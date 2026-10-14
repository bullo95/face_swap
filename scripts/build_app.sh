#!/bin/zsh
# Construit dist/FaceCam.app (modèles inclus). Signature ad hoc pour que macOS accepte l'accès caméra.
set -e
cd "$(dirname "$0")/.."
rm -rf build dist
export PYINSTALLER_CONFIG_DIR="$PWD/build/pyinstaller-cache"
.venv/bin/python scripts/compile_coreml_models.py
.venv/bin/pyinstaller --noconfirm facecam.spec
codesign --force --deep --sign - dist/FaceCam.app
# Signature ad hoc : l'empreinte change à chaque build et macOS refuse alors la caméra en silence.
# On réinitialise l'autorisation pour que la demande soit reposée au prochain lancement.
tccutil reset Camera fr.bruno.facecam >/dev/null 2>&1 || true
echo "OK : dist/FaceCam.app"
