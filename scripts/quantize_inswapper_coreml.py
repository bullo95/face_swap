"""Produit les deux modèles CoreML natifs d'inswapper à partir du fp16 (cible macOS 15) :
  - models/inswapper_128_int8.mlpackage : poids int8 par canal  (Qualité, 48 ms, écart 0,003)
  - models/inswapper_128_p6.mlpackage   : palette 6 bits groupée (Rapide, 39 ms, écart 0,009)
et compare fidélité / latence Neural Engine (nécessite scikit-learn pour la palette).

Usage : .venv/bin/python scripts/quantize_inswapper_coreml.py [models/inswapper_128.mlpackage]
"""
import os, sys, time, glob
import numpy as np, cv2
import coremltools as ct
import coremltools.optimize.coreml as cto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "/private/tmp/claude-502/-Volumes-WKLD-Projets-deepfake/c7a2d22b-ed0e-459c-b8a9-2b8a10717923/scratchpad/out"

src = sys.argv[1] if len(sys.argv) > 1 else "models/inswapper_128.mlpackage"
dst = "models/inswapper_128_int8.mlpackage"
dst_fast = "models/inswapper_128_p6.mlpackage"

from app.engine import FaceEngine, Settings
from insightface.utils import face_align
eng = FaceEngine(Settings()); eng.load()
sp = os.path.dirname(os.path.dirname(cv2.__file__))
imgs = sorted(glob.glob(os.path.join(sp, "insightface", "data", "images", "*.jpg")) + glob.glob(os.path.join(sp, "insightface", "data", "images", "*.png")))
faces, latents = [], []
for p in imgs:
    im = cv2.imread(p)
    if im is None: continue
    for f in eng.analyser.get(im):
        aimg, _ = face_align.norm_crop2(im, f.kps, 128)
        faces.append(cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True).astype(np.float32))
        lat = f.normed_embedding.reshape(1, -1).astype(np.float32) @ eng.swapper.emap; lat /= np.linalg.norm(lat)
        latents.append(lat.astype(np.float32))
samples = [{"target": faces[i], "source": latents[(i + 1) % len(latents)]} for i in range(len(faces))]
print(f"{len(samples)} échantillons de test")

base = ct.models.MLModel(src, compute_units=ct.ComputeUnit.CPU_ONLY)
ref_m = ct.models.MLModel(src, compute_units=ct.ComputeUnit.CPU_AND_NE)
refs = [ref_m.predict(s)["output"] for s in samples]

def evaluate(label, m):
    diffs = [np.abs(m.predict(s)["output"] - r).mean() for s, r in zip(samples, refs)]
    for _ in range(5): m.predict(samples[0])
    l = []
    for _ in range(30):
        t = time.perf_counter(); m.predict(samples[0]); l.append(time.perf_counter() - t)
    print(f"{label:22s}: écart moyen vs fp16 {np.mean(diffs):.4f} max {np.max(diffs):.4f} | latence NE méd {np.median(l)*1000:.0f} ms")
    out = m.predict(samples[0])["output"][0].transpose(1, 2, 0)[:, :, ::-1]
    cv2.imwrite(f"{OUT}/quant_{label.split()[0]}.png", np.clip(out * 255, 0, 255).astype(np.uint8))

evaluate("fp16 (référence)", ref_m)
variants = {
    "int8 par canal": lambda: cto.linear_quantize_weights(base, cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel"))),
    "palette6 groupe32": lambda: cto.palettize_weights(base, cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=6, granularity="per_grouped_channel", group_size=32))),
}
results = {}
for label, mk in variants.items():
    try:
        t0 = time.time(); q = mk(); el = time.time() - t0
        path = f"{OUT}/{label.split()[0]}.mlpackage"; q.save(path)
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f"  ({label} quantifié en {el:.0f}s)"); evaluate(label, m); results[label] = path
    except Exception as e:
        print(f"{label}: échec {repr(e)[:160]}")
# le modèle int8 est la version retenue
import shutil
np.save("models/inswapper_emap.npy", eng.swapper.emap.astype(np.float32)); print("écrit : models/inswapper_emap.npy")
for key, target in (("int8 par canal", dst), ("palette6 groupe32", dst_fast)):
    if key in results:
        if os.path.exists(target): shutil.rmtree(target)
        shutil.copytree(results[key], target); print("écrit :", target)
