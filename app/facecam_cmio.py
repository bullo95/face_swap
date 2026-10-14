"""Wrapper ctypes de libfacecam_cmio.dylib : pousse des images BGRA dans la caméra virtuelle « FaceCam Camera »
(extension CoreMediaIO du projet FaceCamCamera/)."""
from __future__ import annotations

import ctypes
import os
import sys
from typing import List, Tuple

import numpy as np

DEVICE_UID = "7A3F5C21-FACE-4CAM-B0D1-000000000001"  # kDeviceUID de l'extension


def _lib_path() -> str:
    cands = []
    if getattr(sys, "frozen", False):
        cands += [os.path.join(getattr(sys, "_MEIPASS", ""), "libfacecam_cmio.dylib"),
                  os.path.join(os.path.dirname(sys.executable), "..", "Frameworks", "libfacecam_cmio.dylib")]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands.append(os.path.join(root, "FaceCamCamera", "Client", "libfacecam_cmio.dylib"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("libfacecam_cmio.dylib introuvable (compiler FaceCamCamera/Client)")


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_lib_path())
        _lib.facecam_cmio_open.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
        _lib.facecam_cmio_open.restype = ctypes.c_int32
        _lib.facecam_cmio_push.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32]
        _lib.facecam_cmio_push.restype = ctypes.c_int32
        _lib.facecam_cmio_close.argtypes = [ctypes.c_int32]
        _lib.facecam_cmio_list.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        _lib.facecam_cmio_list.restype = ctypes.c_int32
    return _lib


def list_devices() -> List[Tuple[str, str]]:
    buf = ctypes.create_string_buffer(16384)
    lib().facecam_cmio_list(buf, 16384)
    out = []
    for line in buf.value.decode("utf-8", "replace").splitlines():
        if "|" in line:
            uid, name = line.split("|", 1)
            out.append((uid, name))
    return out


def is_installed() -> bool:
    return any(uid == DEVICE_UID for uid, _ in list_devices())


class FaceCamVirtualCamera:
    """Envoie des images BGR (OpenCV) vers FaceCam Camera."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width, self.height = width, height
        h = lib().facecam_cmio_open(DEVICE_UID.encode(), width, height)
        if h <= 0:
            raise RuntimeError({-1: "FaceCam Camera introuvable : installe l'extension avec l'app « FaceCam Camera ».",
                                -2: "flux d'entrée introuvable", -3: "file d'attente CoreMediaIO indisponible",
                                -4: "impossible de démarrer le flux"}.get(h, f"erreur CoreMediaIO {h}"))
        self._h = h
        self._bgra = np.empty((height, width, 4), dtype=np.uint8)
        self._bgra[:, :, 3] = 255

    def send(self, frame_bgr: np.ndarray):
        import cv2
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        self._bgra[:, :, :3] = frame_bgr
        return lib().facecam_cmio_push(self._h, self._bgra.ctypes.data, self.width * 4)

    def close(self):
        if getattr(self, "_h", 0):
            lib().facecam_cmio_close(self._h)
            self._h = 0
