#!/bin/zsh
# Construit dist/FaceCam.app (modèles inclus). Signature ad hoc pour que macOS accepte l'accès caméra.
set -e
cd "$(dirname "$0")/.."
rm -rf build dist
export PYINSTALLER_CONFIG_DIR="$PWD/build/pyinstaller-cache"
.venv/bin/pyinstaller --noconfirm facecam.spec
codesign --force --deep --sign - dist/FaceCam.app
echo "OK : dist/FaceCam.app"
