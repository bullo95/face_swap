"""Sortie vers la caméra virtuelle « FaceCam Camera » (extension CoreMediaIO du projet FaceCamCamera/)."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class VirtualCamera:
    def __init__(self):
        self._cam = None
        self.width = self.height = 0

    @property
    def active(self) -> bool:
        return self._cam is not None

    @staticmethod
    def installed() -> bool:
        try:
            from .facecam_cmio import is_installed
            return is_installed()
        except Exception:  # noqa: BLE001
            return False

    def start(self, width: int, height: int, fps: int = 30):
        from .facecam_cmio import FaceCamVirtualCamera, is_installed

        self.stop()
        if not is_installed():
            raise RuntimeError(
                "« FaceCam Camera » n'est pas installée. Ouvre l'app « FaceCam Camera » (dossier Applications), "
                "clique « Installer l'extension » et autorise-la dans Réglages Système › Général › "
                "Ouverture de session et extensions › Extensions de caméra."
            )
        # l'extension diffuse en 1280x720 : on envoie ce format quelle que soit la capture
        self._cam = FaceCamVirtualCamera(1280, 720)
        self.width, self.height = 1280, 720
        log.info("Caméra virtuelle FaceCam Camera : %dx%d", self.width, self.height)
        return "FaceCam Camera"

    def send(self, frame_bgr: np.ndarray):
        if self._cam is None:
            return
        self._cam.send(frame_bgr)

    def stop(self):
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:  # noqa: BLE001
                pass
        self._cam = None
