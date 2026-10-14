"""Sortie vers la caméra virtuelle OBS (extension CoreMediaIO) via pyvirtualcam."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VirtualCamera:
    def __init__(self):
        self._cam = None
        self.width = self.height = 0

    @property
    def active(self) -> bool:
        return self._cam is not None

    def start(self, width: int, height: int, fps: int = 30):
        import pyvirtualcam
        from pyvirtualcam import PixelFormat

        self.stop()
        last_err: Optional[Exception] = None
        # macOS 14+ : extension système OBS ; plus ancien : plugin DAL.
        for backend in ("obs", None):
            try:
                self._cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=PixelFormat.BGR, backend=backend)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                self._cam = None
        if self._cam is None:
            raise RuntimeError(
                "Caméra virtuelle indisponible. Lance OBS une fois, clique « Démarrer la caméra virtuelle » "
                "et autorise l'extension dans Réglages Système > Général > Ouverture de session et extensions. "
                f"Détail : {last_err}"
            )
        self.width, self.height = width, height
        log.info("Caméra virtuelle : %s (%dx%d@%d)", self._cam.device, width, height, fps)
        return self._cam.device

    def send(self, frame_bgr: np.ndarray):
        if self._cam is None:
            return
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        self._cam.send(frame_bgr)

    def stop(self):
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:  # noqa: BLE001
                pass
        self._cam = None
