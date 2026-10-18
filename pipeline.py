# -*- coding: utf-8 -*-
"""
FORTEYE pipeline: local (offline) detection of cashier-discipline violations.

Everything runs on this machine: Ultralytics YOLO weights and a local VLM
(Qwen2-VL-7B or Florence-2) are loaded from disk. No network calls at inference.
"""
from __future__ import annotations

import base64
import bisect
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2
import numpy as np

LOG = logging.getLogger("forteye.pipeline")
ProgressCb = Callable[[float, str], None]

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIRS = ("models", "weights")  # model files are looked up in both folders


def _search_roots() -> List[Path]:
    roots: List[Path] = []
    for r in (Path.cwd(), BASE_DIR, BASE_DIR.parent):
        if r not in roots:
            roots.append(r)
    return roots


def resolve_path(name: str, want_dir: bool = False) -> Optional[str]:
    """Find a weights file / model folder no matter where the server was started from.

    Tries the path as given, then relative to cwd, to this file and its parent, and finally
    inside ./models and ./weights (by base name). Returns None when nothing is found.
    """
    if not name:
        return None
    ok = (lambda q: q.is_dir()) if want_dir else (lambda q: q.is_file())
    p = Path(name).expanduser()
    if p.is_absolute():
        return str(p) if ok(p) else None
    base = Path(name).name
    for root in _search_roots():
        for cand in [root / p] + [root / d / base for d in MODEL_DIRS]:
            if ok(cand):
                return str(cand.resolve())
    return None


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def _env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except (TypeError, ValueError):
        LOG.warning("bad value %s=%r, using %r", name, raw, default)
        return default


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    sample_fps: float = field(default_factory=lambda: _env("FORTEYE_SAMPLE_FPS", 1.5, float))
    max_side: int = field(default_factory=lambda: _env("FORTEYE_MAX_SIDE", 960, int))
    jpeg_quality: int = 80
    pose_weights: str = field(default_factory=lambda: _env("FORTEYE_YOLO_POSE", "yolo11n-pose.pt"))
    det_weights: str = field(default_factory=lambda: _env("FORTEYE_YOLO_DET", "yolo11s.pt"))
    custom_weights: str = field(default_factory=lambda: _env("FORTEYE_YOLO_CUSTOM", "custom_yolo.pt"))
    yolo_conf: float = 0.30
    det_conf: float = field(default_factory=lambda: _env("FORTEYE_DET_CONF", 0.25, float))  # objects: drinks/bags are small
    yolo_imgsz: int = 640
    device: str = field(default_factory=lambda: _env("FORTEYE_DEVICE", ""))
    vlm: str = field(default_factory=lambda: _env("FORTEYE_VLM", "qwen2vl").lower())
    vlm_path: str = field(default_factory=lambda: _env("FORTEYE_VLM_PATH", "Qwen2-VL-7B-Instruct"))
    vlm_4bit: bool = field(default_factory=lambda: _flag("FORTEYE_VLM_4BIT", False))
    vlm_max_pixels: int = 640 * 28 * 28
    vlm_max_new_tokens: int = 220
    strict: bool = field(default_factory=lambda: _flag("FORTEYE_STRICT", True))
    zones_file: str = field(default_factory=lambda: _env("FORTEYE_ZONES", ""))
    vlm_every_sec: float = field(default_factory=lambda: _env("FORTEYE_VLM_EVERY", 4.0, float))
    vlm_dense_sec: float = 8.0
    vlm_max_frames: int = field(default_factory=lambda: _env("FORTEYE_VLM_MAX_FRAMES", 500, int))
    motion_thr: float = 4.0
    time_budget_sec: float = field(default_factory=lambda: _env("FORTEYE_TIME_BUDGET", 0.0, float))
    min_obs: int = 3
    min_score: float = 0.5
    absent_min_sec: float = 4.0
    door_open_sec: float = 5.0
    safe_open_sec: float = 10.0
    item_min_sec: float = 6.0
    unattended_sec: float = 4.0
    session_min_sec: float = 3.0
    desk_clear_sec: float = 8.0
    switch_gap_sec: float = 8.0
    after_session_sec: float = 6.0
    lock_min_sec: float = 1.3
    enable_lock_rule: bool = field(default_factory=lambda: _flag("FORTEYE_LOCK_RULE", True))
    merge_sec: float = 15.0
    max_per_code: int = 5
    evidence_n: int = field(default_factory=lambda: _env("FORTEYE_EVIDENCE_N", 3, int))
    evidence_step_sec: float = 1.5


# --------------------------------------------------------------------------- #
# NBRK rules catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    code: str
    type: str
    severity: str
    source: str
    why: str


RULES: Dict[str, Rule] = {r.code: r for r in [
    Rule("R01", "Отсутствие проверки личности клиента (KYC)", "high",
         "Ст. 5 Закона РК «О ПОД/ФТ»; Правила № 231",
         "Нарушение требования о надлежащей проверке клиента (KYC) до совершения операций."),
    Rule("R02", "Отсутствие проверки инкассаторской службы", "high",
         "Правила № 231, Глава 4",
         "Нарушение порядка приёма/передачи ценностей инкассаторам: кассовый работник обязан проверить удостоверения."),
    Rule("R03", "Приём инкассации без пересчёта и сверки с данными АБС", "high",
         "Правила № 231, Параграф 2/4",
         "Нарушение процедуры приёма ценностей от инкассаторов и сверки фактических сумм."),
    Rule("R04", "Открытая дверь кассового узла", "high",
         "Правила Нацбанка РК; Правила № 231",
         "Дверь кассового узла должна быть закрыта: открытая дверь допускает доступ посторонних в кассовую зону."),
    Rule("R05", "Сейф оставлен открытым / ключ в замке сейфа", "high",
         "Правила № 231",
         "Сейфы и металлические шкафы закрываются на ключ сразу после извлечения/вложения денег."),
    Rule("R06", "Недостаточное количество оборотов ключа в сейфе/кассе", "mid",
         "Внутренние регламенты БВУ",
         "Нарушение правил эксплуатации замковых устройств и хранения ключей."),
    Rule("R07", "Уход с рабочего места с ценностями вне сейфа", "high",
         "Правила № 231",
         "Кассовый работник обязан убирать деньги, штампы и печати в сейф при любой отлучке."),
    Rule("R08", "Уход с рабочего места с включённым ПК", "mid",
         "Правила № 231, п. 22; Постановление Правления НБ РК № 48",
         "Обязательна блокировка учётной записи при оставлении ПК."),
    Rule("R09", "Начало работы с новым клиентом без завершения предыдущей операции", "mid",
         "Правила № 231",
         "Обслуживание нового клиента до окончания расчётов предыдущего запрещено."),
    Rule("R10", "Смешение наличности разных клиентов", "high",
         "Правила № 231",
         "Смешение денежной наличности разных клиентов на рабочем столе кассира запрещено."),
    Rule("R11", "Пропуск проверки подлинности банкнот (УФ/ИК/на просвет)", "high",
         "Правила № 231, п. 9",
         "Кассир обязан проверить подлинность банкнот приборами перед оприходованием."),
    Rule("R12", "Личные вещи, верхняя одежда, еда или напитки в кассовой зоне", "low",
         "Правила внутреннего трудового распорядка",
         "В кассовой кабине запрещено хранить личные вещи, еду и напитки."),
    Rule("R13", "Инкассаторский мешок без присмотра", "high",
         "Правила № 231, Глава 4",
         "Нарушение требований к инкассации и сохранности ценностей."),
    Rule("R14", "Присвоение денежных средств", "high",
         "Правила № 231, ст. 189 УК РК",
         "Присвоение средств квалифицируется по ст. 189 УК РК."),
    Rule("R15", "Деньги на столе после операции / работа без пересчёта", "mid",
         "Правила № 231, п. 9",
         "Обязательны приборы проверки подлинности и сплошной пересчёт."),
    Rule("R16", "Проведение операций без присутствия клиента", "high",
         "Правила документального оформления БВУ РК",
         "Все кассовые операции совершаются в строгом присутствии клиента."),
]}


def list_rules() -> List[Dict[str, str]]:
    return [{"code": r.code, "type": r.type, "severity": r.severity, "source": r.source} for r in RULES.values()]


@dataclass
class Person:
    box: Tuple[float, float, float, float]
    conf: float
    wrists: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class Det:
    label: str
    conf: float
    box: Tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2


@dataclass
class Snap:
    i: int
    t: float
    persons: List[Person]
    dets: List[Det]
    obs: Optional[Dict[str, Any]] = None
    f: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    code: str
    t: float
    seen: str
    score: float = 0.7


@dataclass
class Run:
    t0: float
    t1: float
    n: int

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


def mmss(t: float) -> str:
    m, s = divmod(int(max(0.0, t)), 60)
    return f"{m:02d}:{s:02d}"


def _shrink(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return frame
    k = max_side / float(s)
    return cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def probe_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return float(n / fps) if fps > 1 and n > 0 else 0.0
    finally:
        cap.release()


def iter_frames(path: str, sample_fps: float, max_side: int) -> Iterator[Tuple[float, np.ndarray]]:
    vr = None
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
    except Exception:
        pass

    if vr is not None:
        fps = float(vr.get_avg_fps()) or 29.0
        n = len(vr)
        step = max(1.0, fps / sample_fps)
        idxs = sorted({int(round(i * step)) for i in range(int(n / step) + 1) if int(round(i * step)) < n})
        for k in range(0, len(idxs), 16):
            chunk = idxs[k:k + 16]
            batch = vr.get_batch(chunk).asnumpy()
            for fi, rgb in zip(chunk, batch):
                yield fi / fps, _shrink(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), max_side)
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:
        fps = 29.0
    step = max(1.0, fps / sample_fps)
    nxt, idx = 0.0, 0
    try:
        while cap.grab():
            if idx >= nxt - 1e-6:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx / fps, _shrink(frame, max_side)
                nxt += step
            idx += 1
    finally:
        cap.release()


COCO_TO_ITEM = {
    "handbag": "bag", "backpack": "bag", "suitcase": "bag",
    "bottle": "drink", "cup": "drink", "wine glass": "drink",
    "sandwich": "food", "banana": "food", "apple": "food", "orange": "food", "pizza": "food",
    "donut": "food", "cake": "food", "hot dog": "food", "broccoli": "food", "carrot": "food",
}
CUSTOM_ALIASES = {
    "cash": {"cash", "money", "banknote", "banknotes", "bill", "bills"},
    "safe": {"safe", "vault"},
    "safe_open": {"safe_open", "open_safe", "safe-open"},
    "key": {"key", "keys"},
    "hand": {"hand", "hands"},
    "door_open": {"door_open", "open_door"},
    "collector_bag": {"collector_bag", "cash_bag", "incasso_bag", "inkas_bag"},
}


def _canon_custom(name: str) -> str:
    n = name.strip().lower().replace(" ", "_")
    for canon, aliases in CUSTOM_ALIASES.items():
        if n in aliases:
            return canon
    return n


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pose = None
        self.det = None
        self.custom = None
        self.custom_known: Set[str] = set()
        self.paths: Dict[str, str] = {}
        self.loaded = False

    def _open(self, yolo_cls, path: str, required: bool):
        found = resolve_path(path)
        if found is None:
            msg = (f"YOLO weights not found: '{path}'. Looked in cwd, {BASE_DIR} and their "
                   f"{'/'.join(MODEL_DIRS)} folders.")
            if required and self.cfg.strict:
                raise FileNotFoundError(msg)
            LOG.warning(msg)
            return None
        LOG.info("loading YOLO weights: %s", found)
        self.paths[path] = found
        return yolo_cls(found)

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            if self.cfg.strict:
                raise RuntimeError("ultralytics is not installed") from exc
            self.loaded = True
            return
        self.pose = self._open(YOLO, self.cfg.pose_weights, required=True)
        self.det = self._open(YOLO, self.cfg.det_weights, required=False)
        if self.cfg.custom_weights:
            # custom_yolo.pt is picked up automatically; it is mandatory only when set explicitly via env
            self.custom = self._open(YOLO, self.cfg.custom_weights, required=bool(os.getenv("FORTEYE_YOLO_CUSTOM")))
            if self.custom is not None:
                self.custom_known = {_canon_custom(n) for n in self.custom.names.values()}
        self.loaded = True

    def __call__(self, frame: np.ndarray, t: float = 0.0) -> Tuple[List[Person], List[Det]]:
        persons: List[Person] = []
        dets: List[Det] = []
        kw: Dict[str, Any] = dict(imgsz=self.cfg.yolo_imgsz, conf=self.cfg.yolo_conf, verbose=False)
        if self.cfg.device:
            kw["device"] = self.cfg.device
        if self.pose is not None:
            r = self.pose.predict(frame, **kw)[0]
            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes.xyxyn.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                kp = r.keypoints
                kxy = kp.xyn.cpu().numpy() if kp is not None else None
                kcf = kp.conf.cpu().numpy() if (kp is not None and kp.conf is not None) else None
                for i in range(len(boxes)):
                    wr: List[Tuple[float, float]] = []
                    if kxy is not None:
                        for k in (9, 10):
                            x, y = kxy[i][k]
                            c = float(kcf[i][k]) if kcf is not None else 1.0
                            if c > 0.35 and (x > 0 or y > 0):
                                wr.append((float(x), float(y)))
                    persons.append(Person(tuple(float(v) for v in boxes[i]), float(confs[i]), wr))
        for model, is_custom in ((self.det, False), (self.custom, True)):
            if model is None:
                continue
            r = model.predict(frame, **(kw if is_custom else dict(kw, conf=self.cfg.det_conf)))[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for box, c, k in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                raw = r.names[int(k)]
                label = _canon_custom(raw) if is_custom else COCO_TO_ITEM.get(raw)
                if label:
                    dets.append(Det(label, float(c), tuple(float(v) for v in box)))
        return persons, dets


OBS_BOOL = ("cashier", "client", "door_open", "safe_open", "key_in_safe", "locking",
            "cash_desk", "valuables", "handling", "counting", "authcheck", "idcheck", "cash_in",
            "cash_out", "pocketing", "collector", "collector_docs", "cbag", "cbag_alone", "pc_use")
ITEM_NAMES = ("outerwear", "bag", "food", "drink")

OBS_PROMPT = """Top-down CCTV frame of a bank cashier workstation. Report only what is clearly visible; when unsure answer 0.
Answer with ONE JSON object and no other text. Use 1 (yes) or 0 (no):
cashier: a bank employee is at the cashier workplace
client: a customer is at the service window/counter in front of the cashier
door_open: the cashier booth / cash node door is open or ajar
safe_open: a safe or metal cabinet door is open
key_in_safe: a key sticks in the lock of a safe/cabinet
locking: the cashier is turning a key to lock a safe/cabinet right now
cash_desk: banknotes or cash bundles lie on the desk or counter
valuables: cash, stamps, seals or documents lie on the desk
handling: the cashier touches, sorts or moves banknotes
counting: the cashier counts banknotes (by hand or with a machine)
authcheck: banknotes are checked with a UV/IR detector or held against the light
idcheck: an identity document is shown to or examined by the cashier
cash_in: the customer hands banknotes to the cashier
cash_out: the cashier hands cash to the customer
pocketing: the cashier puts banknotes into a pocket, clothing or a personal bag
collector: uniformed cash-in-transit collectors are present
collector_docs: the collectors' ID / power of attorney / schedule card are being examined
cbag: a cash-in-transit bag is visible
cbag_alone: such a bag lies with nobody next to it
pc_use: the cashier is typing or using the computer
screen: "on" | "off" | "na"
items: list from ["outerwear","bag","food","drink"]
JSON:"""


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def parse_observation(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    data: Optional[Dict[str, Any]] = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict):
        data = {k: v.strip('"') for k, v in re.findall(r'"?([a-z_]+)"?\s*:\s*("?[A-Za-z0-9]+"?)', text)}
    obs: Dict[str, Any] = {k: _truthy(data[k]) for k in OBS_BOOL if k in data}
    scr = str(data.get("screen", "")).strip().lower()
    if scr in {"on", "off", "na"}:
        obs["screen"] = scr
    items = data.get("items")
    if isinstance(items, list):
        obs["items"] = {str(i).lower() for i in items if str(i).lower() in ITEM_NAMES}
    return obs or None


class VLM:
    name = "none"
    enabled = False

    def load(self) -> None:
        pass

    def observe(self, frame: np.ndarray, t: float = 0.0) -> Optional[Dict[str, Any]]:
        return None


class QwenVLM(VLM):
    name = "qwen2vl"
    enabled = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch  # type: ignore
        from transformers import AutoProcessor  # type: ignore
        try:
            from transformers import Qwen2VLForConditionalGeneration as ModelCls  # type: ignore
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls  # type: ignore
        path = resolve_path(self.cfg.vlm_path, want_dir=True)
        if path is None:
            raise FileNotFoundError(
                f"Qwen2-VL folder not found: '{self.cfg.vlm_path}' (looked in cwd, {BASE_DIR} and their "
                f"{'/'.join(MODEL_DIRS)} folders). Download it: huggingface-cli download Qwen/Qwen2-VL-7B-Instruct "
                f"--local-dir models/Qwen2-VL-7B-Instruct, or set FORTEYE_VLM_PATH, or FORTEYE_VLM=none.")
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else torch.float16
        kw: Dict[str, Any] = dict(local_files_only=True, torch_dtype=dtype)
        if self.cfg.vlm_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
            kw["device_map"] = "auto"
        else:
            kw["device_map"] = {"": self.cfg.device or ("cuda" if cuda else "cpu")}
        self.model = ModelCls.from_pretrained(path, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(
            path, local_files_only=True, min_pixels=128 * 28 * 28, max_pixels=self.cfg.vlm_max_pixels)

    def observe(self, frame: np.ndarray, t: float = 0.0) -> Optional[Dict[str, Any]]:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OBS_PROMPT}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.vlm_max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        txt = self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return parse_observation(txt)


def make_vlm(cfg: Config) -> VLM:
    if cfg.vlm == "qwen2vl":
        return QwenVLM(cfg)
    return VLM()


class Anchor:
    def __init__(self, gx: int = 24, gy: int = 14):
        self.gx, self.gy = gx, gy
        self.h = np.zeros((gy, gx), np.float32)
        self.n = 0
        self._c: Optional[Tuple[float, float]] = None

    def add(self, persons: List[Person]) -> None:
        for p in persons:
            cx, cy = p.center
            self.h[min(self.gy - 1, max(0, int(cy * self.gy))), min(self.gx - 1, max(0, int(cx * self.gx)))] += 1
            self.n += 1
        self._c = None

    def ready(self) -> bool:
        return self.n >= 8

    def center(self) -> Optional[Tuple[float, float]]:
        if not self.ready():
            return None
        if self._c is None:
            sm = cv2.blur(self.h, (3, 3))
            gy, gx = np.unravel_index(int(np.argmax(sm)), sm.shape)
            self._c = ((gx + 0.5) / self.gx, (gy + 0.5) / self.gy)
        return self._c

    def near(self, persons: List[Person]) -> bool:
        c = self.center()
        if c is None:
            return True
        return any(abs(p.center[0] - c[0]) <= 0.2 and abs(p.center[1] - c[1]) <= 0.26 for p in persons)


class Workspace:
    def __init__(self, cfg: Config, anchor: Anchor):
        self.anchor = anchor
        self.poly: Optional[np.ndarray] = None
        if cfg.zones_file and os.path.isfile(cfg.zones_file):
            try:
                with open(cfg.zones_file, encoding="utf-8") as fh:
                    pts = json.load(fh).get("cash_zone")
                if pts and len(pts) >= 3:
                    self.poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            except (OSError, ValueError) as exc:
                LOG.warning("cannot read zones file: %s", exc)

    def ready(self) -> bool:
        return self.poly is not None or self.anchor.ready()

    def contains(self, x: float, y: float, scale: float = 1.0) -> bool:
        if self.poly is not None:
            return cv2.pointPolygonTest(self.poly, (float(x), float(y)), False) >= 0
        c = self.anchor.center()
        if c is None:
            return False
        return ((x - c[0]) / (0.17 * scale)) ** 2 + ((y - c[1]) / (0.24 * scale)) ** 2 <= 1.0


class Selector:
    HOT = ("safe_open", "key_in_safe", "locking", "pocketing", "cbag_alone", "door_open")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last = -1e9
        self.dense_until = -1e9
        self.prev_n: Optional[int] = None

    def should_run(self, t: float, persons: List[Person], anchor: Anchor, motion: float) -> bool:
        n = len(persons)
        if self.prev_n is not None and n != self.prev_n:
            self.dense_until = max(self.dense_until, t + self.cfg.vlm_dense_sec)
        self.prev_n = n
        gap = t - self.last
        if gap >= self.cfg.vlm_every_sec or t <= self.dense_until:
            return True
        if anchor.ready() and not anchor.near(persons) and gap >= 2.0:
            return True
        return motion >= self.cfg.motion_thr and gap >= 1.5

    def mark(self, t: float) -> None:
        self.last = t

    def feedback(self, t: float, obs: Dict[str, Any]) -> None:
        if any(obs.get(k) for k in self.HOT):
            self.dense_until = max(self.dense_until, t + 6.0)


STRONG = {"safe_open", "door_open", "cash_desk", "key_in_safe", "cbag", "cbag_alone"}


def _near(a: Det, b: Det, pad: float = 0.05) -> bool:
    return (b.box[0] - pad <= a.cx <= b.box[2] + pad) and (b.box[1] - pad <= a.cy <= b.box[3] + pad)


def build_facts(snaps: List[Snap], ws: Workspace, det: Any, cfg: Optional[Config] = None) -> None:
    cfg = cfg or Config()
    known: Set[str] = set(getattr(det, "custom_known", set()))
    have_pose = getattr(det, "pose", None) is not None
    have_det = getattr(det, "det", None) is not None
    pose_ok = have_pose and ws.ready()
    vlm_active = any(s.obs for s in snaps)
    # The VLM only runs every few seconds, so "a client is at the window" is smoothed over a time window.
    client_ts = sorted(s.t for s in snaps if s.obs and s.obs.get("client"))
    win = max(6.0, 1.5 * cfg.vlm_every_sec)

    def within(ts: List[float], t: float) -> bool:
        j = bisect.bisect_left(ts, t - win)
        return j < len(ts) and ts[j] <= t + win

    for s in snaps:
        y: Dict[str, Any] = {}
        near = within(client_ts, s.t)   # the VLM saw a client at the window around this moment
        if pose_ok:
            inside = [p for p in s.persons if ws.contains(*p.center)]
            y["cashier"] = len(inside) >= 1
        if have_det:
            # YOLO is authoritative for objects. A bag next to a client at the window is the client's own.
            y["items"] = {d.label for d in s.dets
                          if d.label in ("bag", "drink", "food") and not (near and d.label == "bag")
                          and ws.contains(d.cx, d.cy, 1.2)}
        by = lambda lab: [d for d in s.dets if d.label == lab and d.conf >= 0.4]
        if "safe_open" in known:
            y["safe_open"] = bool(by("safe_open"))
        if "door_open" in known:
            y["door_open"] = bool(by("door_open"))
        if "cash" in known:
            cash = by("cash")
            y["cash_desk"] = bool(cash)
            wr = [w for p in s.persons for w in p.wrists]
            y["handling"] = any(c.box[0] - .03 <= x <= c.box[2] + .03 and c.box[1] - .03 <= yy <= c.box[3] + .03
                                for c in cash for x, yy in wr)
        if "key" in known and "safe" in known:
            y["key_in_safe"] = any(_near(k, sf, 0.02) for k in by("key") for sf in by("safe"))
        if "collector_bag" in known:
            bags = by("collector_bag")
            y["cbag"] = bool(bags)
            y["cbag_alone"] = any(all(abs(p.center[0] - b.cx) + abs(p.center[1] - b.cy) > 0.3 for p in s.persons)
                                  for b in bags)
        f: Dict[str, Any] = {}
        for k, v in y.items():
            if k in STRONG or k in ("cashier", "items") or not vlm_active:
                f[k] = v
        o = s.obs
        if o:
            for k in OBS_BOOL:
                if k not in o or (k in STRONG and k in y):
                    continue
                f[k] = (bool(f.get(k)) or bool(o[k])) if k == "cashier" else bool(o[k])
            if "screen" in o:
                f["screen"] = o["screen"]
            if "items" in o and "items" not in y:  # VLM guesses objects only when there is no detector
                f["items"] = set(o["items"])
        s.f = f


class Timeline:
    def __init__(self, snaps: List[Snap], max_gap: float):
        self.snaps = snaps
        self.max_gap = max_gap
        self._cache: Dict[str, List[Tuple[float, Any]]] = {}

    def vals(self, key: str, t0: float = -1e18, t1: float = 1e18) -> List[Tuple[float, Any]]:
        if key not in self._cache:
            self._cache[key] = [(s.t, s.f[key]) for s in self.snaps if s.f.get(key) is not None]
        return [(t, v) for t, v in self._cache[key] if t0 <= t <= t1]

    def n(self, key: str, t0: float, t1: float) -> int:
        return len(self.vals(key, t0, t1))

    def any(self, key: str, t0: float, t1: float) -> Optional[bool]:
        v = self.vals(key, t0, t1)
        return None if not v else any(bool(x) for _, x in v)

    def times_true(self, key: str, t0: float = -1e18, t1: float = 1e18) -> List[float]:
        return [t for t, v in self.vals(key, t0, t1) if v]

    def runs(self, key: str, min_n: int = 2, min_dur: float = 0.0,
             pred: Optional[Callable[[Any], bool]] = None) -> List[Run]:
        pred = pred or bool
        out: List[List[float]] = []
        cur: Optional[List[float]] = None
        for t, v in self.vals(key):
            if pred(v):
                if cur is not None and t - cur[1] <= self.max_gap:
                    cur[1] = t
                    cur[2] += 1
                else:
                    if cur is not None:
                        out.append(cur)
                    cur = [t, t, 1]
            elif cur is not None:
                out.append(cur)
                cur = None
        if cur is not None:
            out.append(cur)
        return [Run(a, b, int(n)) for a, b, n in out if n >= min_n and (b - a) >= min_dur]


def _merge_intervals(runs: List[Run], gap: float) -> List[Tuple[float, float]]:
    merged: List[List[float]] = []
    for r in runs:
        if merged and r.t0 - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], r.t1)
        else:
            merged.append([r.t0, r.t1])
    return [(a, b) for a, b in merged]


def _overlaps(ivs: List[Tuple[float, float]], t0: float, t1: float, pad: float = 0.0) -> bool:
    return any(a - pad <= t1 and t0 <= b + pad for a, b in ivs)


def _sc(base: float, n: int, k: float = 0.05, cap: float = 0.9) -> float:
    return min(cap, base + k * n)


# --------------------------------------------------------------------------- #
# Правила с безопасной логикой оценки пропусков данных
# --------------------------------------------------------------------------- #

def r_no_client_id(tl: Timeline, S, C, cfg: Config) -> List[Event]:
    ev = []
    for a, b in S:
        ts = sorted(set(tl.times_true("cash_in", a, b + 2) + tl.times_true("handling", a, b + 2)
                        + tl.times_true("cash_out", a, b + 2)))
        if not ts:
            continue
        t_c = ts[0]
        n = tl.n("idcheck", a - 3, t_c + 2)
        # ИСПРАВЛЕНИЕ: строгая проверка объема выборки вместо небезопасного `is not True`
        if n >= cfg.min_obs and tl.any("idcheck", a - 3, t_c + 2) is False:
            ev.append(Event("R01", t_c, f"Клиент у окна с {mmss(a)}; кассир приступил к операции с наличными, "
                            f"но проверка документа не просматривается (проверено кадров: {n}).", _sc(0.5, n)))
    return ev


def r_no_collector_id(tl, S, C, cfg) -> List[Event]:
    ev = []
    for a, b in C:
        n = tl.n("collector_docs", a - 2, b)
        if n >= cfg.min_obs and tl.any("collector_docs", a - 2, b) is False:
            t0 = min(tl.times_true("cbag", a, b) + tl.times_true("handling", a, b) + [a])
            ev.append(Event("R02", t0, f"Инкассаторы в зоне с {mmss(a)} по {mmss(b)}; проверка документов не зафиксирована (кадров: {n}).",
                            _sc(0.5, n)))
    return ev


def r_collection_not_counted(tl, S, C, cfg) -> List[Event]:
    ev = []
    for a, b in C:
        n = tl.n("counting", a, b + 30)
        if n >= cfg.min_obs and tl.any("counting", a, b + 30) is False:
            ev.append(Event("R03", b, f"Приём ценностей завершён ({mmss(a)}–{mmss(b)}), пересчёт суммы не зафиксирован (кадров: {n}).", _sc(0.5, n)))
    return ev


def r_access(tl, S, C, cfg) -> List[Event]:
    ev = []
    for r in tl.runs("door_open", 2, cfg.door_open_sec):
        ev.append(Event("R04", r.t0, f"Дверь кассового узла остаётся открытой не менее {r.dur:.0f} с.", _sc(0.55, r.n)))
    return ev


def r_safe(tl, S, C, cfg) -> List[Event]:
    ev = []
    for r in tl.runs("safe_open", 2):
        absent = sum(1 for _, v in tl.vals("cashier", r.t0, r.t1) if not v) >= 2
        long_open = r.dur >= cfg.safe_open_sec and not _overlaps(C, r.t0, r.t1)
        if absent or long_open:
            why = "пока кассир отсутствует" if absent else f"в течение {r.dur:.0f} с"
            ev.append(Event("R05", r.t0, f"Сейф остаётся открытым {why}.", _sc(0.55, r.n)))
    for r in tl.runs("key_in_safe", 2, 3.0):
        ev.append(Event("R05", r.t0, f"В замке сейфа остаётся ключ не менее {r.dur:.0f} с.", _sc(0.55, r.n)))
    return ev


def r_lock_turns(tl, S, C, cfg) -> List[Event]:
    if not cfg.enable_lock_rule:
        return []
    ev = []
    vals = tl.vals("locking")
    for r in tl.runs("locking", 1):
        tb = max((t for t, v in vals if t < r.t0 and not v), default=None)
        ta = min((t for t, v in vals if t > r.t1 and not v), default=None)
        if tb is None or ta is None or r.t0 - tb > 1.6 or ta - r.t1 > 1.6:
            continue
        if r.dur < cfg.lock_min_sec:
            ev.append(Event("R06", r.t0, f"Запирание сейфа заняло около {max(r.dur, 0.5):.1f} с — недостаточно оборотов ключа.", 0.55))
    return ev


def r_leaving(tl, S, C, cfg) -> List[Event]:
    ev = []
    for r in tl.runs("cashier", 2, cfg.absent_min_sec, pred=lambda v: not v):
        if _overlaps(C, r.t0, r.t1):
            continue
        val = sorted(set(tl.times_true("valuables", r.t0, r.t1) + tl.times_true("cash_desk", r.t0, r.t1)))
        if len(val) >= 2:
            ev.append(Event("R07", val[0], f"Кассир покинул рабочее место, оставив ценности на столе вне сейфа.", _sc(0.5, len(val))))
        scr = [t for t, v in tl.vals("screen", r.t0, r.t1) if v == "on"]
        if len(scr) >= 2:
            ev.append(Event("R08", scr[0], f"Кассир покинул рабочее место, экран ПК остаётся включённым.", _sc(0.5, len(scr))))
    return ev


def r_switch_client(tl, S, C, cfg) -> List[Event]:
    ev = []
    for (a1, b1), (a2, b2) in zip(S, S[1:]):
        if a2 - b1 > cfg.switch_gap_sec:
            continue
        cleared = any(not v for _, v in tl.vals("cash_desk", b1, a2 + 2))
        if cleared or not (tl.any("cash_desk", b1 - 3, b1 + 1) and tl.any("cash_desk", a2, a2 + 6)):
            continue
        cnt = tl.n("counting", a2, b2)
        mixed = bool(tl.any("cash_in", a2, b2)) and cnt >= cfg.min_obs - 1 and tl.any("counting", a2, b2) is False
        if mixed:
            ev.append(Event("R10", a2, f"Деньги предыдущего клиента на столе, наличность нового принята без пересчёта — смешение наличности.", 0.7))
        else:
            ev.append(Event("R09", a2, f"Начато обслуживание нового клиента, деньги предыдущего не убраны в сейф.", 0.65))
    return ev


def r_authenticity(tl, S, C, cfg) -> List[Event]:
    ev = []
    for a, b in S:
        recv = tl.times_true("cash_in", a, b + 2)
        if not recv:
            continue
        n = tl.n("authcheck", a, b + 4)
        if n >= cfg.min_obs and tl.any("authcheck", a, b + 4) is False:
            ev.append(Event("R11", recv[0] + 2, f"Наличные приняты, но проверка подлинности (УФ/ИК) не зафиксирована (кадров: {n}).", _sc(0.5, n)))
    return ev


def r_personal_items(tl, S, C, cfg) -> List[Event]:
    ev = []
    ru = {"outerwear": "верхняя одежда", "bag": "личная сумка", "food": "еда", "drink": "напитки"}
    for r in tl.runs("items", 3, cfg.item_min_sec):
        if _overlaps(C, r.t0, r.t1, 3.0):
            continue
        names: Set[str] = set()
        for _, v in tl.vals("items", r.t0, r.t1):
            names |= set(v)
        ev.append(Event("R12", r.t0, f"В рабочей зоне кассира обнаружены запрещенные предметы: {', '.join(ru[x] for x in sorted(names))}.", _sc(0.5, r.n, 0.04, 0.85)))
    return ev


def r_unattended_bag(tl, S, C, cfg) -> List[Event]:
    return [Event("R13", r.t0, f"Инкассаторский мешок лежит без присмотра не менее {r.dur:.0f} с.", _sc(0.55, r.n))
            for r in tl.runs("cbag_alone", 2, cfg.unattended_sec)]


def r_pocketing(tl, S, C, cfg) -> List[Event]:
    ts = tl.times_true("pocketing")
    groups: List[List[float]] = []
    for t in ts:
        if groups and t - groups[-1][-1] <= 8.0:
            groups[-1].append(t)
        else:
            groups.append([t])
    return [Event("R14", g[0], f"Кассир убирает банкноты в карман или личную сумку.", _sc(0.55, len(g), 0.1)) for g in groups if len(g) >= 2]


def r_desk(tl, S, C, cfg) -> List[Event]:
    ev = []
    for i, (a, b) in enumerate(S):
        nxt = S[i + 1][0] if i + 1 < len(S) else 1e18
        w0, w1 = b + cfg.desk_clear_sec, min(b + cfg.desk_clear_sec + 8.0, nxt - 1.0)
        if w1 > w0:
            ts = tl.times_true("cash_desk", w0, w1)
            if len(ts) >= 2 and tl.any("cashier", w0, w1):
                ev.append(Event("R15", ts[0], f"Операция завершена, но деньги остаются на столе.", _sc(0.5, len(ts))))
        handled = tl.times_true("cash_in", a, b) + tl.times_true("handling", a, b)
        n = tl.n("counting", a, b + 4)
        if handled and n >= cfg.min_obs and tl.any("counting", a, b + 4) is False:
            ev.append(Event("R15", b, f"Работа с наличными клиента велась без пересчёта (кадров: {n}).", _sc(0.5, n)))
    return ev


def r_no_client(tl, S, C, cfg) -> List[Event]:
    if not S:
        return []
    def quiet(t: float) -> bool:
        client_vals = tl.vals("client", t - 3, t + 3)
        explicit_no_client = len(client_vals) >= cfg.min_obs and all(not v for _, v in client_vals)
        return (not _overlaps(S, t, t, cfg.after_session_sec)) and not _overlaps(C, t, t, 3.0) and explicit_no_client

    ts = sorted({t for k in ("counting", "cash_out", "cash_in") for t in tl.times_true(k) if quiet(t)})
    groups: List[List[float]] = []
    for t in ts:
        if groups and t - groups[-1][-1] <= 8.0:
            groups[-1].append(t)
        else:
            groups.append([t])
    return [Event("R16", g[0], f"Кассир выполняет операции с наличными при отсутствии клиента у окна.", _sc(0.5, len(g))) for g in groups if len(g) >= 2]


ALL_RULES = (r_no_client_id, r_no_collector_id, r_collection_not_counted, r_access, r_safe, r_lock_turns,
             r_leaving, r_switch_client, r_authenticity, r_personal_items, r_unattended_bag, r_pocketing,
             r_desk, r_no_client)


def finalize(events: List[Event], cfg: Config) -> List[Event]:
    events = sorted((e for e in events if e.score >= cfg.min_score), key=lambda e: e.t)
    out: List[Event] = []
    for e in events:
        dup = next((o for o in out if o.code == e.code and abs(o.t - e.t) <= cfg.merge_sec), None)
        if dup is not None:
            dup.score = max(dup.score, e.score)
            continue
        out.append(e)
    r10 = [e.t for e in out if e.code == "R10"]
    out = [e for e in out if not (e.code == "R09" and any(abs(e.t - x) <= 10 for x in r10))]
    counts: Dict[str, int] = {}
    kept = []
    for e in out:
        counts[e.code] = counts.get(e.code, 0) + 1
        if counts[e.code] <= cfg.max_per_code:
            kept.append(e)
    return kept


def build_sessions(tl: Timeline, cfg: Config) -> List[Tuple[float, float]]:
    vals = tl.vals("client")
    merged: List[List[float]] = []
    for r in tl.runs("client", 1):
        if merged:
            falses = sum(1 for t, v in vals if merged[-1][1] < t < r.t0 and not v)
            if falses < 2 and r.t0 - merged[-1][1] <= 2 * cfg.vlm_every_sec + 1:
                merged[-1][1] = r.t1
                continue
        merged.append([r.t0, r.t1])
    return [(a, b) for a, b in merged if b - a >= cfg.session_min_sec]


def run_rules(snaps: List[Snap], cfg: Config) -> List[Event]:
    tl = Timeline(snaps, max_gap=max(10.0, 2.5 * cfg.vlm_every_sec))
    coll = _merge_intervals(tl.runs("collector", 2), 15.0)
    sessions = [(a, b) for a, b in build_sessions(tl, cfg) if not _overlaps(coll, a, b, 5.0)]
    events: List[Event] = []
    for rule in ALL_RULES:
        try:
            events += rule(tl, sessions, coll, cfg)
        except Exception:
            LOG.exception("rule %s failed", rule.__name__)
    return finalize(events, cfg)


CYAN, PINK = (252, 211, 98), (84, 21, 179)


def render_evidence(jpeg: bytes, snap: Snap, code: str) -> str:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    for p in snap.persons:
        x1, y1, x2, y2 = p.box
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), CYAN, 2)
        for x, y in p.wrists:
            cv2.circle(img, (int(x * w), int(y * h)), 5, PINK, -1)
    for d in snap.dets:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), PINK, 2)
        cv2.putText(img, d.label, (int(x1 * w) + 3, max(14, int(y1 * h) - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, PINK, 1,
                    cv2.LINE_AA)
    cap = f"{code}  T+{mmss(snap.t)}"
    (tw, th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, .6, 2)
    cv2.rectangle(img, (0, 0), (tw + 14, th + 14), (0, 0, 0), -1)
    cv2.putText(img, cap, (7, th + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, CYAN, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def pick_snaps(times: List[float], t: float, cfg: Config) -> List[int]:
    offs = [0.0]
    k = 1
    while len(offs) < max(1, cfg.evidence_n):
        offs += [-k * cfg.evidence_step_sec, k * cfg.evidence_step_sec]
        k += 1
    idxs: List[int] = []
    for o in offs[:max(1, cfg.evidence_n)]:
        tt = t + o
        j = bisect.bisect_left(times, tt)
        cand = [c for c in (j - 1, j) if 0 <= c < len(times)]
        best = min(cand, key=lambda c: abs(times[c] - tt))
        if best not in idxs:
            idxs.append(best)
    return sorted(idxs)


@dataclass
class Collected:
    snaps: List[Snap]
    cache: Dict[int, bytes]
    anchor: Anchor
    duration: float


class Engine:
    def __init__(self, cfg: Optional[Config] = None, detector: Optional[Any] = None, vlm: Optional[VLM] = None):
        self.cfg = cfg or Config()
        self.detector = detector or Detector(self.cfg)
        self.vlm = vlm or make_vlm(self.cfg)
        self._lock = threading.RLock()
        self._loaded = False
        self.vlm_error: Optional[str] = None

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self.detector.load()
            try:
                self.vlm.load()
            except FileNotFoundError as exc:  # model folder missing: run on YOLO only instead of failing the job
                LOG.warning("VLM disabled, analysis continues with YOLO only: %s", exc)
                self.vlm_error = str(exc)
                self.vlm = VLM()
            except Exception as exc:
                if self.cfg.strict:
                    raise
                self.vlm_error = str(exc)
                self.vlm = VLM()
            self._loaded = True

    def capabilities(self) -> Dict[str, Any]:
        return {"vlm": self.vlm.name, "vlm_error": self.vlm_error,
                "yolo_pose": getattr(self.detector, "pose", None) is not None,
                "yolo_objects": getattr(self.detector, "det", None) is not None,
                "custom_classes": sorted(getattr(self.detector, "custom_known", [])),
                "weights": dict(getattr(self.detector, "paths", {})),
                "sample_fps": self.cfg.sample_fps, "loaded": self._loaded}

    def _collect(self, path: str, progress: ProgressCb) -> Collected:
        cfg = self.cfg
        duration = probe_duration(path)
        anchor, sel = Anchor(), Selector(cfg)
        snaps: List[Snap] = []
        cache: Dict[int, bytes] = {}
        prev_gray = None
        calls = 0
        t_start = time.time()
        for i, (t, frame) in enumerate(iter_frames(path, cfg.sample_fps, cfg.max_side)):
            persons, dets = self.detector(frame, t)
            anchor.add(persons)
            gray = cv2.cvtColor(cv2.resize(frame, (80, 45)), cv2.COLOR_BGR2GRAY)
            motion = float(np.mean(cv2.absdiff(gray, prev_gray))) if prev_gray is not None else 0.0
            prev_gray = gray
            snap = Snap(i, t, persons, dets)
            in_budget = not (cfg.time_budget_sec and time.time() - t_start > 0.7 * cfg.time_budget_sec)
            if self.vlm.enabled and calls < cfg.vlm_max_frames and in_budget and sel.should_run(t, persons, anchor, motion):
                try:
                    obs = self.vlm.observe(frame, t)
                except Exception:
                    obs = None
                calls += 1
                sel.mark(t)
                if obs:
                    snap.obs = obs
                    sel.feedback(t, obs)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            if ok:
                cache[i] = buf.tobytes()
            snaps.append(snap)
            frac = min(1.0, t / duration) if duration > 0 else 0.0
            progress(0.03 + 0.85 * frac, f"Кадр {mmss(t)}")
        return Collected(snaps, cache, anchor, duration)

    def analyze(self, path: str, progress: Optional[ProgressCb] = None) -> Dict[str, Any]:
        progress = progress or (lambda f, s: None)
        with self._lock:
            progress(0.0, "Загрузка моделей")
            self.load()
            col = self._collect(path, progress)
            if not col.snaps:
                raise RuntimeError("no frames could be decoded from the video")
            progress(0.90, "Проверка правил НБ РК")
            build_facts(col.snaps, Workspace(self.cfg, col.anchor), self.detector, self.cfg)
            events = run_rules(col.snaps, self.cfg)
            progress(0.95, "Сбор доказательств")
            times = [s.t for s in col.snaps]
            violations = []
            for e in events:
                rule = RULES[e.code]
                frames = []
                for j in pick_snaps(times, e.t, self.cfg):
                    if j in col.cache:
                        frames.append({"timestamp_sec": round(col.snaps[j].t, 2),
                                       "image_base64": render_evidence(col.cache[j], col.snaps[j], e.code)})
                violations.append({
                    "type": rule.type,
                    "timecode_sec": round(e.t, 2),
                    "description": f"{e.seen} Почему это нарушение: {rule.why} Основание: {rule.source}.",
                    "evidence_frames": frames,
                })
            progress(1.0, "Готово")
            return {"verdict": "violation" if violations else "no_violation", "violations": violations}


_ENGINE: Optional[Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Engine()
        return _ENGINE


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if len(sys.argv) < 2:
        raise SystemExit("usage: python pipeline.py video.mov [--full]")
    res = get_engine().analyze(sys.argv[1], progress=lambda f, s: print(f"[{f:4.0%}] {s}", file=sys.stderr))
    if "--full" not in sys.argv:
        for v in res["violations"]:
            v["evidence_frames"] = [{"timestamp_sec": x["timestamp_sec"], "image_base64": "..."} for x in v["evidence_frames"]]
    print(json.dumps(res, ensure_ascii=False, indent=2))