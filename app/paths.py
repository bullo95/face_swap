import os
import sys


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bundled_models_dir():
    """Dossier models/ dans une app PyInstaller : sys._MEIPASS (Contents/Frameworks) ou Contents/Resources."""
    if not getattr(sys, "frozen", False):
        return None
    candidates = [
        os.path.join(getattr(sys, "_MEIPASS", ""), "models"),
        os.path.join(os.path.dirname(sys.executable), "..", "Resources", "models"),
        os.path.join(os.path.dirname(sys.executable), "models"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return None


MODELS_DIR = os.environ.get("FACECAM_MODELS") or _bundled_models_dir() or os.path.join(project_root(), "models")
INSIGHTFACE_ROOT = os.path.join(MODELS_DIR, "insightface")
INSWAPPER_PATH = os.path.join(MODELS_DIR, "inswapper_128.onnx")
INSWAPPER_FP16_PATH = os.path.join(MODELS_DIR, "inswapper_128_fp16.onnx")
GFPGAN_PATH = os.path.join(MODELS_DIR, "gfpgan_1.4.onnx")
INSWAPPER_COREML_PATH = os.path.join(MODELS_DIR, "inswapper_128_int8.mlpackage")  # CoreML natif, poids int8 (qualité)
INSWAPPER_COREML_FAST_PATH = os.path.join(MODELS_DIR, "inswapper_128_p6.mlpackage")  # CoreML natif, palette 6 bits (rapide)
