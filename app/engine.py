"""Moteur de face swap : détection (buffalo_l), swap (inswapper_128), amélioration (GFPGAN 1.4 ONNX)."""
from __future__ import annotations

import logging
import os

# BLAS mono-thread : un produit matriciel 1x512·512x512 par image faisait tourner 8 threads en attente active.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import threading
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from .paths import GFPGAN_PATH, INSIGHTFACE_ROOT, INSWAPPER_COREML_FAST_PATH, INSWAPPER_COREML_PATH, INSWAPPER_FP16_PATH, INSWAPPER_PATH

log = logging.getLogger(__name__)

# Gabarit FFHQ 512 (5 points) utilisé par GFPGAN, coordonnées normalisées.
FFHQ_512 = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.58646219, 0.72490465],
    ],
    dtype=np.float32,
) * 512.0


def session_options() -> ort.SessionOptions:
    """Pools de threads CPU sobres : CoreML fait le gros du calcul, les threads ORT ne doivent pas tourner
    en attente active (mesuré : 20 threads à 100 % CPU sinon, puis étranglement thermique)."""
    so = ort.SessionOptions()
    spin = os.environ.get("FACECAM_SPIN", "0")
    so.add_session_config_entry("session.intra_op.allow_spinning", spin)
    so.add_session_config_entry("session.inter_op.allow_spinning", spin)
    so.intra_op_num_threads = int(os.environ.get("FACECAM_THREADS", "2"))
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


COMPUTE_UNITS = {  # libellé UI -> option CoreML
    # Mesuré dans l'app, caméra ouverte : Neural Engine seul = 89 ms stable ; Auto (ANE+GPU) = 63-238 ms,
    # car la part GPU subit la concurrence de la capture et de l'affichage. À vide, Auto est plus rapide (48 ms).
    "Neural Engine seul (recommandé)": "CPUAndNeuralEngine",
    "Auto (Neural Engine + GPU)": "ALL",
    "GPU seul": "CPUAndGPU",
}
DEFAULT_COMPUTE_UNITS = "CPUAndNeuralEngine"

SWAP_QUALITIES = {  # libellé UI -> clé
    "Qualité (int8, 48 ms)": "quality",
    "Rapide (palette 6 bits, 39 ms)": "fast",
}
DEFAULT_SWAP_QUALITY = "quality"

MODES = {  # libellé UI -> clé
    "Remplacer le visage (swap)": "swap",
    "Flouter le visage": "blur",
    "Pixelliser le visage": "pixelate",
    "Masquer le visage (noir)": "mask",
}


def providers(compute_units: str = "ALL") -> list:
    """CoreML si disponible (Apple Silicon), sinon CPU."""
    avail = ort.get_available_providers()
    out = []
    if "CoreMLExecutionProvider" in avail:
        # Mesuré sur M4 : MLProgram = 49 ms (inswapper fp16) / 194 ms (GFPGAN), NeuralNetwork bien plus lent.
        out.append(("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": compute_units}))
    out.append("CPUExecutionProvider")
    return out


@dataclass
class Settings:
    enhance: bool = False          # GFPGAN après le swap
    enhance_blend: float = 0.8     # poids du visage amélioré (0..1)
    many_faces: bool = False       # remplacer tous les visages ou seulement le plus grand
    det_size: int = 320            # résolution du détecteur (320 rapide, 640 précis)
    compute_units: str = DEFAULT_COMPUTE_UNITS  # unité CoreML pour le swapper (voir COMPUTE_UNITS)
    swap_quality: str = DEFAULT_SWAP_QUALITY    # "quality" (int8) ou "fast" (palette 6 bits), modèles CoreML natifs
    mode: str = "swap"                          # voir MODES ; les modes d'anonymisation traitent tous les visages
    anonymize_strength: float = 0.6             # 0..1 : force du flou / taille des pixels


class FaceEngine:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._lock = threading.Lock()
        self.source_face = None
        self._latent = None  # vecteur d'identité projeté (calculé une fois par photo source)
        self.analyser = None
        self.swapper = None
        self.enhancer: Optional[ort.InferenceSession] = None
        self._enh_input = None
        self.stats = {"detect": 0.0, "swap": 0.0, "paste": 0.0, "enhance": 0.0, "n": 0}

    # ---------- chargement ----------
    def load(self, progress=None):
        import insightface
        from insightface.app import FaceAnalysis

        prov = providers()
        log.info("onnxruntime providers: %s", prov)
        self._swap_path = None

        if progress:
            progress("Chargement du détecteur de visages (buffalo_l)…")
        self.analyser = FaceAnalysis(
            name="buffalo_l",
            root=INSIGHTFACE_ROOT,
            providers=prov,
            sess_options=session_options(),
            allowed_modules=["detection", "recognition"],
        )
        s = self.settings.det_size
        self.analyser.prepare(ctx_id=0, det_size=(s, s))

        if progress:
            progress("Chargement du modèle de swap (inswapper_128)…")
        # fp16 (49 ms sur CoreML) de préférence, sinon fp32 (121 ms).
        swap_path = INSWAPPER_FP16_PATH if os.path.exists(INSWAPPER_FP16_PATH) else INSWAPPER_PATH
        if not os.path.exists(swap_path):
            raise FileNotFoundError(f"Modèle manquant : {INSWAPPER_PATH}")
        self._swap_path = swap_path
        self.swapper = self._make_swapper(swap_path, self.settings.compute_units)

        if os.path.exists(GFPGAN_PATH):
            if progress:
                progress("Chargement de l'améliorateur (GFPGAN 1.4)…")
            self.enhancer = ort.InferenceSession(GFPGAN_PATH, sess_options=session_options(), providers=prov)
            self._enh_input = self.enhancer.get_inputs()[0].name
        else:
            log.warning("GFPGAN absent (%s) : amélioration désactivée", GFPGAN_PATH)

        self._self_test()
        if progress:
            progress("Prêt")

    def _self_test(self):
        """Journalise les fournisseurs réellement actifs et le temps d'une inférence à vide (chauffe CoreML)."""
        import time
        try:
            log.info("providers détecteur: %s", self.analyser.det_model.session.get_providers())
            log.info("providers swapper: %s", self.swapper.session.get_providers())
            if self.enhancer is not None:
                log.info("providers enhancer: %s", self.enhancer.get_providers())
            x = np.zeros((1, 3, 128, 128), np.float32); lat = np.zeros((1, 512), np.float32)
            names = [i.name for i in self.swapper.session.get_inputs()]
            feed = {names[0]: x, names[1]: lat}
            self.swapper.session.run(None, feed)
            t = time.perf_counter()
            for _ in range(5):
                self.swapper.session.run(None, feed)
            log.info("swapper inférence à vide: %.0f ms", (time.perf_counter() - t) / 5 * 1000)
        except Exception:  # noqa: BLE001
            log.exception("self-test")

    def _make_swapper(self, onnx_path: str, cu: str, quality: Optional[str] = None):
        """CoreML natif si disponible : int8 (qualité, 48 ms) ou palette 6 bits (rapide, 39 ms) sur Neural Engine seul,
        mesuré M4 ; sinon ONNX Runtime (79 ms)."""
        import insightface
        quality = quality or self.settings.swap_quality
        candidates = [INSWAPPER_COREML_FAST_PATH, INSWAPPER_COREML_PATH] if quality == "fast" else [INSWAPPER_COREML_PATH, INSWAPPER_COREML_FAST_PATH]
        if not os.environ.get("FACECAM_NO_COREML"):
            for path in candidates:
                if not os.path.isdir(path):
                    continue
                try:
                    from .coreml_swapper import CoreMLSwapper
                    sw = CoreMLSwapper(path, INSWAPPER_PATH if os.path.exists(INSWAPPER_PATH) else onnx_path, cu)
                    log.info("swapper: CoreML natif %s (%s, %s)", os.path.basename(path), cu, quality)
                    return sw
                except Exception:  # noqa: BLE001
                    log.exception("CoreML natif %s indisponible", os.path.basename(path))
        log.info("swapper: ONNX Runtime %s (%s)", os.path.basename(onnx_path), cu)
        return insightface.model_zoo.get_model(onnx_path, providers=providers(cu), sess_options=session_options())

    def set_compute_units(self, cu: str):
        """Recharge uniquement la session du swapper avec une autre unité CoreML (~1 s)."""
        if cu == self.settings.compute_units or self._swap_path is None:
            return
        new = self._make_swapper(self._swap_path, cu)
        with self._lock:
            self.swapper = new
            self.settings.compute_units = cu
        log.info("swapper rechargé avec MLComputeUnits=%s", cu)

    def set_swap_quality(self, quality: str):
        """Bascule Qualité (int8) / Rapide (palette 6 bits) en rechargeant le swapper (~1 s)."""
        if quality == self.settings.swap_quality or self._swap_path is None:
            return
        new = self._make_swapper(self._swap_path, self.settings.compute_units, quality)
        with self._lock:
            self.swapper = new
            self.settings.swap_quality = quality
        log.info("swapper rechargé en mode %s", quality)

    # ---------- visage source ----------
    def set_source_image(self, path: str) -> bool:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image illisible : {path}")
        faces = self.analyser.get(img)
        if not faces:
            self.source_face = None
            return False
        # visage le plus grand
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        latent = faces[0].normed_embedding.reshape((1, -1)).astype(np.float32)
        latent = np.dot(latent, self.swapper.emap)
        latent /= np.linalg.norm(latent)
        with self._lock:
            self.source_face = faces[0]
            self._latent = latent.astype(np.float32)
        return True

    def clear_source(self):
        with self._lock:
            self.source_face = None
            self._latent = None

    def _swap_face(self, frame: np.ndarray, face) -> tuple:
        """Équivalent de INSwapper.get(paste_back=False) avec le vecteur d'identité précalculé."""
        from insightface.utils import face_align
        sw = self.swapper
        aimg, M = face_align.norm_crop2(frame, face.kps, sw.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / sw.input_std, sw.input_size, (sw.input_mean,) * 3, swapRB=True)
        pred = sw.session.run(sw.output_names, {sw.input_names[0]: blob, sw.input_names[1]: self._latent})[0]
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        return bgr_fake, M

    # ---------- traitement d'une image ----------
    def detect(self, frame: np.ndarray) -> list:
        """Étape 1 du pipeline : visages (bbox + 5 points) de l'image."""
        import time
        t0 = time.perf_counter()
        faces = self._detect_only(frame)
        self.stats["detect"] += time.perf_counter() - t0
        return faces

    def process(self, frame: np.ndarray, faces: Optional[list] = None) -> np.ndarray:
        """Traitement d'une image selon le mode. Si `faces` est fourni (détection faite en parallèle), on ne redétecte pas."""
        import time
        st = self.stats
        if self.settings.mode != "swap":
            if faces is None:
                faces = self.detect(frame)
            st["n"] += 1
            if not faces:
                return frame
            t0 = time.perf_counter()
            out = self._anonymize(frame, faces)
            st["paste"] += time.perf_counter() - t0
            return out
        with self._lock:
            src = self.source_face
        if src is None or self._latent is None or self.swapper is None:
            return frame
        if faces is None:
            faces = self.detect(frame)
        st["n"] += 1
        if not faces:
            return frame
        if not self.settings.many_faces:
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            faces = faces[:1]
        out = frame
        for face in faces:
            t0 = time.perf_counter()
            bgr_fake, M = self._swap_face(out, face)
            t1 = time.perf_counter()
            out = self._paste_back(out, bgr_fake, M)
            t2 = time.perf_counter()
            st["swap"] += t1 - t0
            st["paste"] += t2 - t1
            if self.settings.enhance and self.enhancer is not None:
                out = self._enhance_face(out, face.kps)
                st["enhance"] += time.perf_counter() - t2
        return out

    def _anonymize(self, frame: np.ndarray, faces: list) -> np.ndarray:
        """Flou / pixellisation / masque sur chaque visage, dans une ellipse à bords doux (limité à la ROI)."""
        mode = self.settings.mode
        k = float(np.clip(self.settings.anonymize_strength, 0.05, 1.0))
        h, w = frame.shape[:2]
        out = frame.copy()
        for face in faces:
            x0, y0, x1, y1 = face.bbox[:4]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rw, rh = (x1 - x0) * 0.75, (y1 - y0) * 0.85  # demi-axes : ellipse un peu plus large que la bbox
            X0, Y0 = int(max(0, cx - rw - 8)), int(max(0, cy - rh - 8))
            X1, Y1 = int(min(w, cx + rw + 8)), int(min(h, cy + rh + 8))
            if X1 - X0 < 4 or Y1 - Y0 < 4:
                continue
            roi = out[Y0:Y1, X0:X1]
            size = max(x1 - x0, y1 - y0)
            if mode == "blur":
                sigma = max(2.0, size * 0.12 * k)
                fx = cv2.GaussianBlur(roi, (0, 0), sigma)
            elif mode == "pixelate":
                n = int(max(3, round(16 - 12 * k)))  # 16 blocs (léger) -> 4 blocs (fort)
                small = cv2.resize(roi, (max(1, roi.shape[1] * n // max(roi.shape[:2])), max(1, roi.shape[0] * n // max(roi.shape[:2]))), interpolation=cv2.INTER_LINEAR)
                fx = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                fx = np.zeros_like(roi)
            mask = np.zeros(roi.shape[:2], dtype=np.float32)
            cv2.ellipse(mask, (int(cx - X0), int(cy - Y0)), (int(rw), int(rh)), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, size * 0.04))[:, :, None]
            out[Y0:Y1, X0:X1] = (fx * mask + roi * (1.0 - mask)).astype(np.uint8)
        return out

    def pop_stats(self) -> str:
        st = self.stats
        n = max(st["n"], 1)
        line = "detect %.0f ms, swap %.0f ms, paste %.0f ms, enhance %.0f ms (moy. sur %d)" % (
            st["detect"] / n * 1000, st["swap"] / n * 1000, st["paste"] / n * 1000, st["enhance"] / n * 1000, st["n"])
        for k in st:
            st[k] = 0
        return line

    @staticmethod
    def _paste_back(frame: np.ndarray, bgr_fake: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Collage du visage 128x128 limité à sa zone (ROI) : ~10x moins de calculs que le collage
        plein cadre d'insightface, même érosion/flou de masque."""
        h, w = frame.shape[:2]
        n = bgr_fake.shape[0]
        IM = cv2.invertAffineTransform(M)
        corners = np.array([[0, 0], [n, 0], [0, n], [n, n]], dtype=np.float32)
        pts = cv2.transform(corners[None], IM)[0]
        pad = 4
        x0 = int(max(0, np.floor(pts[:, 0].min()) - pad)); y0 = int(max(0, np.floor(pts[:, 1].min()) - pad))
        x1 = int(min(w, np.ceil(pts[:, 0].max()) + pad)); y1 = int(min(h, np.ceil(pts[:, 1].max()) + pad))
        rw, rh = x1 - x0, y1 - y0
        if rw <= 2 or rh <= 2:
            return frame
        IMr = IM.copy()
        IMr[:, 2] -= (x0, y0)  # affine vers le repère de la ROI
        fake_roi = cv2.warpAffine(bgr_fake, IMr, (rw, rh), borderValue=0)
        mask = cv2.warpAffine(np.full((n, n), 255, np.uint8), IMr, (rw, rh), borderValue=0)
        mask[mask > 20] = 255
        ys, xs = np.where(mask == 255)
        if ys.size == 0:
            return frame
        mask_size = int(np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min())))
        k = max(mask_size // 10, 10)
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
        k = max(mask_size // 20, 5)
        mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0)
        alpha = mask.astype(np.float32)[:, :, None] / 255.0
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        blended = alpha * fake_roi.astype(np.float32) + (1.0 - alpha) * roi
        out = frame.copy()
        out[y0:y1, x0:x1] = blended.astype(np.uint8)
        return out

    def _detect_only(self, frame: np.ndarray) -> list:
        """Détection + points clés uniquement (pas de reconnaissance) : ~8 ms économisés par visage."""
        from insightface.app.common import Face
        bboxes, kpss = self.analyser.det_model.detect(frame, max_num=0, metric="default")
        faces = []
        for i in range(bboxes.shape[0]):
            faces.append(Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=bboxes[i, 4]))
        return faces

    # ---------- GFPGAN ONNX ----------
    def _enhance_face(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), FFHQ_512, method=cv2.RANSAC, ransacReprojThreshold=100)
        if M is None:
            return frame
        crop = cv2.warpAffine(frame, M, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        x = crop[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))[None]
        y = self.enhancer.run(None, {self._enh_input: x})[0][0]
        y = np.clip((np.transpose(y, (1, 2, 0)) + 1.0) * 127.5, 0, 255).astype(np.uint8)[:, :, ::-1]
        b = float(self.settings.enhance_blend)
        y = cv2.addWeighted(y, b, crop, 1.0 - b, 0)
        # collage limité à la ROI du visage (même masque adouci qu'avant, mais sans trame complète)
        h, w = frame.shape[:2]
        IM = cv2.invertAffineTransform(M)
        corners = np.array([[0, 0], [512, 0], [0, 512], [512, 512]], dtype=np.float32)
        pts = cv2.transform(corners[None], IM)[0]
        x0 = int(max(0, np.floor(pts[:, 0].min()) - 4)); y0 = int(max(0, np.floor(pts[:, 1].min()) - 4))
        x1 = int(min(w, np.ceil(pts[:, 0].max()) + 4)); y1 = int(min(h, np.ceil(pts[:, 1].max()) + 4))
        rw, rh = x1 - x0, y1 - y0
        if rw <= 2 or rh <= 2:
            return frame
        IMr = IM.copy()
        IMr[:, 2] -= (x0, y0)
        mask = np.zeros((512, 512), dtype=np.float32)
        cv2.rectangle(mask, (32, 32), (480, 480), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 24)
        warped = cv2.warpAffine(y, IMr, (rw, rh), borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(mask, IMr, (rw, rh))[:, :, None]
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        out = frame.copy()
        out[y0:y1, x0:x1] = (warped * alpha + roi * (1.0 - alpha)).astype(np.uint8)
        return out
