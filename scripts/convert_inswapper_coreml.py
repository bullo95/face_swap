"""Convertit inswapper_128.onnx en modèle CoreML natif (mlprogram fp16) et vérifie la fidélité vs ONNX Runtime.

Usage : .venv/bin/python scripts/convert_inswapper_coreml.py [models/inswapper_128.onnx] [models/inswapper_128.mlpackage]
"""
import os, sys, time
import numpy as np
import onnx
import torch
import onnx2torch
import coremltools as ct
import onnxruntime as ort

src = sys.argv[1] if len(sys.argv) > 1 else "models/inswapper_128.onnx"
dst = sys.argv[2] if len(sys.argv) > 2 else "models/inswapper_128.mlpackage"

print("== chargement ONNX", src)
model = onnx.load(src)
# retire les initialiseurs non référencés (emap, buff2fs…) qui perturbent onnx2torch
used = {i for n in model.graph.node for i in n.input}
keep = [t for t in model.graph.initializer if t.name in used]
removed = [t.name for t in model.graph.initializer if t.name not in used]
del model.graph.initializer[:]
model.graph.initializer.extend(keep)
print("initialiseurs retirés :", removed)
print("entrées :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input])

print("== ONNX -> PyTorch")
tm = onnx2torch.convert(model).eval()
target = torch.rand(1, 3, 128, 128)
source = torch.randn(1, 512)
with torch.no_grad():
    out_t = tm(target, source)
print("sortie torch :", tuple(out_t.shape))

print("== trace")
traced = torch.jit.trace(tm, (target, source))

print("== PyTorch -> CoreML mlprogram fp16")
t0 = time.time()
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="target", shape=(1, 3, 128, 128), dtype=np.float32),
            ct.TensorType(name="source", shape=(1, 512), dtype=np.float32)],
    outputs=[ct.TensorType(name="output", dtype=np.float32)],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS14,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
print(f"conversion en {time.time()-t0:.0f}s")
mlmodel.save(dst)
print("sauvegardé :", dst)

print("== fidélité vs ONNX Runtime (CPU, fp32)")
sess = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
rng = np.random.default_rng(0)
tgt = rng.random((1, 3, 128, 128), dtype=np.float32)
lat = rng.standard_normal((1, 512)).astype(np.float32); lat /= np.linalg.norm(lat)
ref = sess.run(None, {names[0]: tgt, names[1]: lat})[0]
loaded = ct.models.MLModel(dst, compute_units=ct.ComputeUnit.CPU_AND_NE)
pred = loaded.predict({"target": tgt, "source": lat})["output"]
diff = np.abs(pred - ref)
print(f"écart max {diff.max():.4f}, moyen {diff.mean():.5f} (échelle 0..1) ; ref min/max {ref.min():.3f}/{ref.max():.3f}")

print("== latence CoreML natif (CPU_AND_NE), 40 inférences")
for _ in range(5): loaded.predict({"target": tgt, "source": lat})
t = time.perf_counter(); n = 40
for _ in range(n): loaded.predict({"target": tgt, "source": lat})
print(f"{(time.perf_counter()-t)/n*1000:.0f} ms par inférence")
