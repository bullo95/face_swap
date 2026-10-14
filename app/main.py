"""FaceCam — interface Qt : prévisualisation, visage source, caméra virtuelle."""
from __future__ import annotations

import logging
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSlider, QStatusBar, QVBoxLayout, QWidget,
)

from .camera import Camera, list_cameras, request_camera_access, set_thread_qos_interactive
from .engine import COMPUTE_UNITS, DEFAULT_COMPUTE_UNITS, DEFAULT_SWAP_QUALITY, MODES, SWAP_QUALITIES, FaceEngine, Settings
from .vcam import VirtualCamera

LOG_PATH = os.path.expanduser("~/Library/Logs/FaceCam.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8")] + ([] if getattr(sys, "frozen", False) else [logging.StreamHandler()]),
)
log = logging.getLogger("facecam")

# App empaquetée (sans console) : capturer stderr natif (avertissements ONNX Runtime / CoreML) dans le journal.
if getattr(sys, "frozen", False):
    try:
        _fd = os.open(LOG_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.dup2(_fd, 2)
        sys.stderr = os.fdopen(2, "a", buffering=1)
    except OSError:
        pass


def _excepthook(exc_type, exc, tb):
    log.critical("Exception non gérée", exc_info=(exc_type, exc, tb))


sys.excepthook = _excepthook


def to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class LoaderThread(QThread):
    progress = Signal(str)
    done = Signal(bool, str)

    def __init__(self, engine: FaceEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.engine.load(progress=self.progress.emit)
            self.done.emit(True, "Modèles chargés")
        except Exception as e:  # noqa: BLE001
            log.exception("Chargement des modèles")
            self.done.emit(False, str(e))


class Worker(QThread):
    """Capture -> swap -> (caméra virtuelle) -> prévisualisation."""
    frame_ready = Signal(QImage, float)
    error = Signal(str)

    def __init__(self, engine: FaceEngine, camera: Camera, vcam: VirtualCamera):
        super().__init__()
        self.engine, self.camera, self.vcam = engine, camera, vcam
        self.swap_enabled = True
        self.mirror_preview = True
        self._running = False

    def run(self):
        """Pipeline : la détection de l'image N+1 tourne pendant le swap de l'image N (thread séparé).
        Le Neural Engine reste ainsi occupé en continu, ce qui évite aussi son sous-cadencement."""
        import resource
        from concurrent.futures import ThreadPoolExecutor
        self._running = True
        set_thread_qos_interactive()
        t_prev, fps = time.perf_counter(), 0.0
        n_frames, n_none, t_log, seq = 0, 0, time.perf_counter(), 0
        ru = resource.getrusage(resource.RUSAGE_SELF); cpu_prev = ru.ru_utime + ru.ru_stime
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detect", initializer=set_thread_qos_interactive)
        pending = None  # (frame, future[faces])
        try:
            while self._running:
                frame, seq = self.camera.read_new(seq)  # attend une nouvelle image : pas de boucle folle
                now = time.perf_counter()
                if now - t_log > 5:
                    ru = resource.getrusage(resource.RUSAGE_SELF); cpu_now = ru.ru_utime + ru.ru_stime
                    cores = (cpu_now - cpu_prev) / (now - t_log); cpu_prev = cpu_now
                    log.info("worker: %.1f fps, %.1f cœurs CPU, mode=%s on=%s enhance=%s many=%s cu=%s | %s",
                             fps, cores, self.engine.settings.mode, self.swap_enabled, self.engine.settings.enhance,
                             self.engine.settings.many_faces, self.engine.settings.compute_units, self.engine.pop_stats())
                    t_log = now
                if frame is None:
                    n_none += 1
                    continue
                n_frames += 1
                if not self.swap_enabled or (self.engine.settings.mode == "swap" and self.engine.source_face is None):
                    pending = None
                    self._emit(frame, fps := self._fps(fps, t_prev)); t_prev = time.perf_counter()
                    continue
                # lance la détection de cette image, puis traite l'image précédente (dont la détection est prête)
                fut = pool.submit(self.engine.detect, frame)
                if pending is None:
                    pending = (frame, fut)
                    continue
                prev_frame, prev_fut = pending
                pending = (frame, fut)
                try:
                    out = self.engine.process(prev_frame, faces=prev_fut.result())
                except Exception as e:  # noqa: BLE001
                    log.exception("process")
                    self.error.emit(str(e))
                    out = prev_frame
                fps = self._fps(fps, t_prev); t_prev = time.perf_counter()
                self._emit(out, fps)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _fps(fps: float, t_prev: float) -> float:
        return 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - t_prev, 1e-6))

    def _emit(self, out, fps: float):
        if self.vcam.active:
            try:
                self.vcam.send(out)
            except Exception as e:  # noqa: BLE001
                self.error.emit(f"Caméra virtuelle : {e}")
                self.vcam.stop()
        preview = cv2.flip(out, 1) if self.mirror_preview else out
        self.frame_ready.emit(to_qimage(preview), fps)

    def stop(self):
        self._running = False
        self.wait(2000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceCam — face swap temps réel")
        self.resize(1200, 720)
        self.engine = FaceEngine(Settings(compute_units=self._default_compute_units(), swap_quality=self._default_swap_quality()))
        self.camera: Optional[Camera] = None
        self.vcam = VirtualCamera()
        self.worker: Optional[Worker] = None

        # --- prévisualisation
        self.preview = QLabel("Chargement des modèles…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background:#111;color:#aaa;font-size:16px;")

        # --- panneau de contrôle (compact : libellé + contrôle sur une ligne)
        panel = QVBoxLayout()
        panel.setSpacing(6)

        g_src = QGroupBox("Visage source")
        l_src = QHBoxLayout(g_src)
        self.src_thumb = QLabel("Aucune\nimage")
        self.src_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.src_thumb.setFixedSize(96, 96)
        self.src_thumb.setStyleSheet("background:#222;color:#888;border-radius:6px;")
        self.btn_src = QPushButton("Choisir une photo…")
        self.btn_src.clicked.connect(self.pick_source)
        self.btn_src_clear = QPushButton("Retirer")
        self.btn_src_clear.clicked.connect(self.clear_source)
        col = QVBoxLayout(); col.addWidget(self.btn_src); col.addWidget(self.btn_src_clear); col.addStretch(1)
        l_src.addWidget(self.src_thumb); l_src.addLayout(col)

        g_cam = QGroupBox("Caméra")
        l_cam = QGridLayout(g_cam)
        self.cam_combo = QComboBox()
        self.res_combo = QComboBox()
        for label, w, h in (("1920x1080", 1920, 1080), ("1280x720", 1280, 720), ("960x540", 960, 540), ("640x480", 640, 480)):
            self.res_combo.addItem(label, (w, h))
        self.res_combo.setCurrentIndex(1)
        self.res_combo.setToolTip("Résolution de capture")
        self.btn_refresh = QPushButton("Rafraîchir")
        self.btn_refresh.clicked.connect(self.refresh_cameras)
        self.btn_cam = QPushButton("Démarrer")
        self.btn_cam.clicked.connect(self.toggle_camera)
        self.btn_vcam = QPushButton("Caméra virtuelle")
        self.btn_vcam.setCheckable(True)
        self.btn_vcam.clicked.connect(self.toggle_vcam)
        self.btn_vcam.setEnabled(False)
        l_cam.addWidget(self.cam_combo, 0, 0, 1, 2)
        l_cam.addWidget(QLabel("Résolution"), 1, 0); l_cam.addWidget(self.res_combo, 1, 1)
        l_cam.addWidget(self.btn_refresh, 2, 0); l_cam.addWidget(self.btn_cam, 2, 1)
        l_cam.addWidget(self.btn_vcam, 3, 0, 1, 2)

        g_opt = QGroupBox("Traitement")
        l_opt = QGridLayout(g_opt)
        self.mode_combo = QComboBox()
        for label in MODES:
            self.mode_combo.addItem(label, MODES[label])
        self.mode_combo.currentIndexChanged.connect(self.apply_options)
        self.sl_anon = QSlider(Qt.Orientation.Horizontal); self.sl_anon.setRange(5, 100); self.sl_anon.setValue(60)
        self.sl_anon.valueChanged.connect(self.apply_options)
        self.cb_swap = QCheckBox("Traitement actif"); self.cb_swap.setChecked(True)
        self.cb_enh = QCheckBox("GFPGAN (lent)")
        self.cb_many = QCheckBox("Tous les visages")
        self.cb_mirror = QCheckBox("Miroir"); self.cb_mirror.setChecked(True)
        for cb in (self.cb_swap, self.cb_enh, self.cb_many, self.cb_mirror):
            cb.toggled.connect(self.apply_options)
        self.sl_blend = QSlider(Qt.Orientation.Horizontal); self.sl_blend.setRange(0, 100); self.sl_blend.setValue(80)
        self.sl_blend.valueChanged.connect(self.apply_options)
        self.cu_combo = QComboBox()
        for label in COMPUTE_UNITS:
            self.cu_combo.addItem(label, COMPUTE_UNITS[label])
        self.cu_combo.setToolTip("Unité CoreML pour le modèle de swap. « Neural Engine seul » évite le partage du GPU avec l'affichage.")
        self.cu_combo.setCurrentIndex(max(0, list(COMPUTE_UNITS.values()).index(self.engine.settings.compute_units)))
        self.cu_combo.currentIndexChanged.connect(self.change_compute_units)
        self.q_combo = QComboBox()
        for label in SWAP_QUALITIES:
            self.q_combo.addItem(label, SWAP_QUALITIES[label])
        self.q_combo.setCurrentIndex(max(0, list(SWAP_QUALITIES.values()).index(self.engine.settings.swap_quality)))
        self.q_combo.setToolTip("Qualité : poids int8, fidèle au modèle d'origine. Rapide : palette 6 bits, ~20 % plus rapide, écart invisible à l'œil.")
        self.q_combo.currentIndexChanged.connect(self.change_swap_quality)
        r = 0
        l_opt.addWidget(QLabel("Mode"), r, 0); l_opt.addWidget(self.mode_combo, r, 1); r += 1
        l_opt.addWidget(QLabel("Flou / pixels"), r, 0); l_opt.addWidget(self.sl_anon, r, 1); r += 1
        l_opt.addWidget(self.cb_swap, r, 0); l_opt.addWidget(self.cb_many, r, 1); r += 1
        l_opt.addWidget(self.cb_mirror, r, 0); l_opt.addWidget(self.cb_enh, r, 1); r += 1
        l_opt.addWidget(QLabel("GFPGAN"), r, 0); l_opt.addWidget(self.sl_blend, r, 1); r += 1
        l_opt.addWidget(QLabel("Modèle"), r, 0); l_opt.addWidget(self.q_combo, r, 1); r += 1
        l_opt.addWidget(QLabel("Calcul"), r, 0); l_opt.addWidget(self.cu_combo, r, 1); r += 1
        l_opt.setColumnStretch(1, 1)
        for combo in (self.cam_combo, self.res_combo, self.mode_combo, self.q_combo, self.cu_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)

        for g in (g_src, g_cam, g_opt):
            panel.addWidget(g)
        panel.addStretch(1)
        side = QWidget()
        side.setLayout(panel)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        root = QWidget()
        lay = QHBoxLayout(root)
        lay.addWidget(self.preview, 1)
        lay.addWidget(scroll)
        self.setCentralWidget(root)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.fps_label = QLabel("")
        self.status.addPermanentWidget(self.fps_label)

        self.set_controls_enabled(False)
        self.loader = LoaderThread(self.engine)
        self.loader.progress.connect(self.status.showMessage)
        self.loader.done.connect(self.on_loaded)
        self.loader.start()

    @staticmethod
    def _default_compute_units() -> str:
        """Préférence macOS : defaults write fr.bruno.facecam ComputeUnits CPUAndNeuralEngine"""
        try:
            from Foundation import NSUserDefaults
            v = NSUserDefaults.standardUserDefaults().stringForKey_("ComputeUnits")
            if v in COMPUTE_UNITS.values():
                log.info("ComputeUnits depuis les préférences : %s", v)
                return str(v)
        except Exception:  # noqa: BLE001
            pass
        return DEFAULT_COMPUTE_UNITS

    @staticmethod
    def _default_swap_quality() -> str:
        """Préférence macOS : defaults write fr.bruno.facecam SwapQuality fast"""
        try:
            from Foundation import NSUserDefaults
            v = NSUserDefaults.standardUserDefaults().stringForKey_("SwapQuality")
            if v in SWAP_QUALITIES.values():
                return str(v)
        except Exception:  # noqa: BLE001
            pass
        return DEFAULT_SWAP_QUALITY

    @staticmethod
    def _save_pref(key: str, value: str):
        try:
            from Foundation import NSUserDefaults
            NSUserDefaults.standardUserDefaults().setObject_forKey_(value, key)
        except Exception:  # noqa: BLE001
            pass

    # ---------- chargement
    def set_controls_enabled(self, on: bool):
        for w in (self.btn_src, self.btn_src_clear, self.btn_cam, self.cam_combo, self.btn_refresh):
            w.setEnabled(on)

    @Slot(bool, str)
    def on_loaded(self, ok: bool, msg: str):
        self.status.showMessage(msg)
        if not ok:
            self.preview.setText("Erreur de chargement :\n" + msg)
            QMessageBox.critical(self, "Modèles", msg)
            return
        self.preview.setText("Choisis une photo source puis démarre la caméra")
        self.refresh_cameras()
        self.set_controls_enabled(True)

    def refresh_cameras(self):
        self.cam_combo.clear()
        self.status.showMessage("Demande d'accès à la caméra…")
        QApplication.processEvents()
        granted = request_camera_access()
        log.info("Accès caméra accordé : %s", granted)
        cams = list_cameras() if granted else []
        log.info("Caméras : %s", cams)
        for uid, label in cams:
            self.cam_combo.addItem(label, uid)
        # caméra intégrée en premier choix si présente
        for i in range(self.cam_combo.count()):
            if "MacBook" in self.cam_combo.itemText(i) or "FaceTime" in self.cam_combo.itemText(i):
                self.cam_combo.setCurrentIndex(i); break
        if not cams:
            self.cam_combo.addItem("Aucune caméra détectée", None)
            self.status.showMessage(
                "Accès caméra refusé ou aucune caméra : Réglages Système › Confidentialité et sécurité › Caméra › FaceCam, puis « Rafraîchir la liste »."
            )
        else:
            self.status.showMessage(f"{len(cams)} caméra(s) détectée(s)")

    # ---------- source
    def pick_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "Photo du visage source", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if not path:
            return
        try:
            ok = self.engine.set_source_image(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Image", str(e))
            return
        if not ok:
            QMessageBox.warning(self, "Image", "Aucun visage détecté dans cette image.")
            return
        pix = QPixmap(path).scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.src_thumb.setPixmap(pix)
        self.status.showMessage(f"Visage source : {os.path.basename(path)}")

    def clear_source(self):
        self.engine.clear_source()
        self.src_thumb.clear()
        self.src_thumb.setText("Aucune\nimage")

    # ---------- options
    def apply_options(self):
        s = self.engine.settings
        s.enhance = self.cb_enh.isChecked()
        s.many_faces = self.cb_many.isChecked()
        s.enhance_blend = self.sl_blend.value() / 100.0
        s.mode = self.mode_combo.currentData()
        s.anonymize_strength = self.sl_anon.value() / 100.0
        if self.worker:
            self.worker.swap_enabled = self.cb_swap.isChecked()
            self.worker.mirror_preview = self.cb_mirror.isChecked()

    def change_swap_quality(self):
        q = self.q_combo.currentData()
        self.status.showMessage("Rechargement du modèle de swap…")
        QApplication.processEvents()
        try:
            self.engine.set_swap_quality(q)
            self._save_pref("SwapQuality", q)
            self.status.showMessage(f"Modèle de swap : {self.q_combo.currentText()}")
        except Exception as e:  # noqa: BLE001
            log.exception("swap quality")
            QMessageBox.warning(self, "Modèle", str(e))

    def change_compute_units(self):
        cu = self.cu_combo.currentData()
        self.status.showMessage("Rechargement du modèle de swap…")
        QApplication.processEvents()
        try:
            self.engine.set_compute_units(cu)
            self._save_pref("ComputeUnits", cu)
            self.status.showMessage(f"Swap sur : {self.cu_combo.currentText()}")
        except Exception as e:  # noqa: BLE001
            log.exception("compute units")
            QMessageBox.warning(self, "Calcul", str(e))

    # ---------- caméra
    def toggle_camera(self):
        if self.worker:
            self.stop_camera()
            return
        idx = self.cam_combo.currentData()
        if idx is None:
            QMessageBox.warning(self, "Caméra", "Aucune caméra disponible.")
            return
        w, h = self.res_combo.currentData()
        log.info("Démarrage caméra id=%s (%s) %dx%d", idx, self.cam_combo.currentText(), w, h)
        try:
            self.camera = Camera(index=idx, width=w, height=h)
            self.camera.start()
            log.info("Caméra ouverte %dx%d", self.camera.width, self.camera.height)
        except Exception as e:  # noqa: BLE001
            log.exception("Ouverture caméra")
            QMessageBox.critical(self, "Caméra", str(e))
            self.camera = None
            return
        self.worker = Worker(self.engine, self.camera, self.vcam)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.error.connect(self.status.showMessage)
        self.apply_options()
        self.worker.start()
        self.btn_cam.setText("Arrêter")
        self.btn_vcam.setEnabled(True)
        self.cam_combo.setEnabled(False)
        self.res_combo.setEnabled(False)
        self.status.showMessage(f"Caméra {self.camera.width}x{self.camera.height}")

    def stop_camera(self):
        if self.vcam.active:
            self.toggle_vcam()
        if self.worker:
            self.worker.stop()
            self.worker = None
        if self.camera:
            self.camera.stop()
            self.camera = None
        self.btn_cam.setText("Démarrer")
        self.btn_vcam.setEnabled(False)
        self.cam_combo.setEnabled(True)
        self.res_combo.setEnabled(True)
        self.fps_label.setText("")

    @Slot(QImage, float)
    def on_frame(self, img: QImage, fps: float):
        pix = QPixmap.fromImage(img).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.preview.setPixmap(pix)
        self.fps_label.setText(f"{fps:.1f} fps")

    # ---------- caméra virtuelle
    def toggle_vcam(self):
        if self.vcam.active:
            self.vcam.stop()
            self.btn_vcam.setChecked(False)
            self.btn_vcam.setText("Caméra virtuelle")
            self.status.showMessage("Caméra virtuelle arrêtée")
            return
        if not self.camera:
            return
        try:
            dev = self.vcam.start(self.camera.width, self.camera.height, 30)
        except Exception as e:  # noqa: BLE001
            self.btn_vcam.setChecked(False)
            QMessageBox.warning(self, "Caméra virtuelle", str(e))
            return
        self.btn_vcam.setChecked(True)
        self.btn_vcam.setText("Caméra virtuelle : active")
        self.status.showMessage(f"Caméra virtuelle active : {dev} — sélectionne-la dans Zoom/Meet/Teams")

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FaceCam")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
