#!/bin/zsh
# Télécharge les modèles dans models/ (≈1,4 Go).
set -e
cd "$(dirname "$0")/.."
mkdir -p models && cd models
FF="https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"
for f in inswapper_128_fp16.onnx inswapper_128.onnx gfpgan_1.4.onnx; do
  [ -f "$f" ] || curl -L --fail --retry 3 -o "$f" "$FF/$f"
done
if [ ! -d insightface/models/buffalo_l ]; then
  curl -L --fail --retry 3 -o buffalo_l.zip "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
  mkdir -p insightface/models/buffalo_l && unzip -o -q buffalo_l.zip -d insightface/models/buffalo_l
fi
ls -lh
