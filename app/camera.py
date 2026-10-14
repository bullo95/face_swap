"""Capture caméra via OpenCV / AVFoundation, dans un thread dédié."""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


def set_thread_qos_interactive():
    """Classe le thread courant en QoS USER_INTERACTIVE : sinon macOS peut le reléguer sur les cœurs efficaces
    et freiner ses requêtes CoreML/Neural Engine (mesuré : swap 130 ms dans l'app contre 48 ms à vide)."""
    try:
        import ctypes
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.pthread_set_qos_class_self_np(ctypes.c_uint(0x21), ctypes.c_int(0))  # QOS_CLASS_USER_INTERACTIVE
    except Exception:  # noqa: BLE001
        pass


def request_camera_access(timeout: float = 60.0) -> bool:
    """Demande l'autorisation caméra à macOS (TCC) et attend la réponse. True si accordée."""
    try:
        import AVFoundation  # PyObjC
    except ImportError:
        return True  # on laisse OpenCV déclencher la demande
    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeVideo)
    if status == 3:  # authorized
        return True
    if status in (1, 2):  # restricted, denied
        return False
    done = threading.Event()
    result = {"ok": False}

    def _cb(granted):
        result["ok"] = bool(granted)
        done.set()

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVFoundation.AVMediaTypeVideo, _cb)
    done.wait(timeout)
    return result["ok"]


def _avfoundation_devices() -> List[Tuple[str, str]]:
    """(uniqueID, nom) des caméras, dans l'ordre AVFoundation `devicesWithMediaType` (celui qu'OpenCV utilise
    pour numéroter les caméras). Cet ordre change selon la caméra par défaut du moment : ne jamais le figer."""
    try:
        import AVFoundation
        devs = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
        return [(str(d.uniqueID()), str(d.localizedName())) for d in devs]
    except Exception:  # noqa: BLE001
        return []


def list_cameras(max_index: int = 6) -> List[Tuple[str, str]]:
    """Énumère les caméras : (uniqueID, nom). Sans ouvrir les caméras (pas de voyant allumé)."""
    devs = _avfoundation_devices()
    if devs:
        return devs
    # repli sans PyObjC : index OpenCV
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            found.append((str(i), f"Caméra {i}"))
        cap.release()
    return found


def opencv_index_for(unique_id: str) -> int:
    """Index OpenCV (AVFoundation) correspondant à l'identifiant unique, calculé au moment de l'ouverture."""
    devs = _avfoundation_devices()
    for i, (uid, _) in enumerate(devs):
        if uid == unique_id:
            return i
    if unique_id.isdigit():
        return int(unique_id)
    raise RuntimeError("Caméra introuvable (débranchée ?) : " + unique_id)


class Camera:
    def __init__(self, index=0, width: int = 1280, height: int = 720, fps: int = 30):
        self.index = index  # int (index OpenCV) ou str (uniqueID AVFoundation, résolu à l'ouverture)
        self.width, self.height, self.fps = width, height, fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        idx = opencv_index_for(self.index) if isinstance(self.index, str) else int(self.index)
        self._cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la caméra {self.index}. Vérifie l'autorisation dans "
                "Réglages Système › Confidentialité et sécurité › Caméra."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self):
        set_thread_qos_interactive()
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._new_frame.notify_all()

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read_new(self, last_seq: int, timeout: float = 0.5):
        """Bloque jusqu'à une image plus récente que last_seq. Retourne (frame, seq) ou (None, last_seq)."""
        with self._lock:
            if self._seq <= last_seq:
                self._new_frame.wait(timeout)
            if self._seq <= last_seq or self._frame is None:
                return None, last_seq
            return self._frame.copy(), self._seq

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
        self._cap = None
