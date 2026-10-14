# -*- mode: python ; coding: utf-8 -*-
# PyInstaller : .venv/bin/pyinstaller facecam.spec
import os, glob
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = os.path.abspath(".")
datas = []
# Modèles (≈1,4 Go) copiés dans Contents/MacOS/models (voir app/paths.py : dossier de l'exécutable).
for f in glob.glob("models/*.onnx"):
    datas.append((f, "models"))
# modèle CoreML natif int8 (dossier .mlpackage) : copié tel quel dans models/
for pkg in ("models/inswapper_128_int8.mlpackage", "models/inswapper_128_p6.mlpackage"):
    if os.path.isdir(pkg):
        datas.append((pkg, pkg))
for f in glob.glob("models/insightface/models/buffalo_l/*.onnx"):
    datas.append((f, "models/insightface/models/buffalo_l"))
datas += collect_data_files("insightface")

hidden = collect_submodules("insightface") + ["pyvirtualcam", "PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtCore", "coremltools", "coremltools.models", "coremltools.proto"]
datas += collect_data_files("coremltools")

a = Analysis(
    ["launcher.py"],
    pathex=[root],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    excludes=["tkinter", "matplotlib", "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="FaceCam",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="FaceCam")
app = BUNDLE(
    coll,
    name="FaceCam.app",
    icon=None,
    bundle_identifier="fr.bruno.facecam",
    info_plist={
        "CFBundleName": "FaceCam",
        "CFBundleDisplayName": "FaceCam",
        "CFBundleShortVersionString": "0.1.0",
        "NSCameraUsageDescription": "FaceCam a besoin de la caméra pour remplacer le visage en direct.",
        # macOS 14+ : désactive par défaut la détection de gestes « Réactions », qui occupe le Neural Engine
        # pendant toute la capture (mesuré : swap 48 ms -> 134 ms caméra ouverte).
        "NSCameraReactionEffectGesturesEnabledDefault": False,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "15.0",  # modèles CoreML palettisés
    },
)
