"""Convertit inswapper_128.onnx en CoreML natif (mlprogram fp16) avec fusion des normalisations d'instance,
puis vérifie la fidélité vs ONNX Runtime et mesure la latence.

Usage : .venv/bin/python scripts/convert_inswapper_coreml.py [--no-fuse] [src.onnx] [dst.mlpackage]
"""
import sys, time
import numpy as np
import onnx, torch, onnx2torch
import coremltools as ct
import onnxruntime as ort
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

args = [a for a in sys.argv[1:] if not a.startswith("--")]
FUSE = "--no-fuse" not in sys.argv
src = args[0] if len(args) > 0 else "models/inswapper_128.onnx"
dst = args[1] if len(args) > 1 else "models/inswapper_128.mlpackage"


def _const_val(var):
    return var.val if var is not None and var.val is not None else None


@register_pass(namespace="custom")
class fuse_adain_instance_norm(AbstractGraphPass):
    """x -> reduce_mean(m) ; d = x - m ; v = reduce_mean(d*d) ; inv = 1/sqrt(v+eps) ; out = d*inv
    ==> instance_norm(x, gamma=1, beta=0, eps). Les gamma/beta dynamiques (AdaIN) restent en aval."""

    def apply(self, prog):
        for f in prog.functions.values():
            self._apply_block(f)

    def _apply_block(self, block):
        fused = 0
        for op in list(block.operations):
            for b in op.blocks:
                self._apply_block(b)
            if op.op_type == "reduce_mean" and self._try(block, op):
                fused += 1
        if fused:
            print(f"  instance_norm fusionnées : {fused}")

    def _single_child(self, var, op_type):
        """Enfant unique de type op_type, en traversant les cast intermédiaires (collectés pour suppression)."""
        while True:
            ch = list(dict.fromkeys(var.child_ops))
            if len(ch) != 1:
                return None
            if ch[0].op_type == "cast":
                self._casts.append(ch[0]); var = ch[0].outputs[0]; continue
            return ch[0] if ch[0].op_type == op_type else None

    @staticmethod
    def _src(var):
        """Remonte les cast pour retrouver la variable d'origine."""
        while var is not None and var.op is not None and var.op.op_type == "cast":
            var = var.op.x
        return var

    def _fail(self, why, rm1):
        if not getattr(self, "_reported", False):
            self._reported = True
            ch = [c.op_type for c in rm1.outputs[0].child_ops]
            print(f"  (diag) premier échec : {why} ; enfants du reduce_mean : {ch}")
        return False

    def _try(self, block, rm1):
        self._casts = []
        x = rm1.x
        axes = _const_val(rm1.axes)
        if axes is None or sorted(int(a) for a in axes) not in ([2, 3], [-2, -1]) or not _const_val(rm1.keep_dims):
            return self._fail(f"axes/keep_dims {axes} {_const_val(rm1.keep_dims)}", rm1)
        sub = self._single_child(rm1.outputs[0], "sub")
        if sub is None or self._src(sub.x) is not self._src(x) or self._src(sub.y) is not rm1.outputs[0]:
            return self._fail("sub", rm1)
        d = sub.outputs[0]
        # d a deux usages : mul(d,d) et le mul final (d*inv)
        children = list(dict.fromkeys(d.child_ops))  # mul(d, d) apparaît deux fois
        if len(children) != 2:
            return self._fail(f"d a {len(children)} usages distincts {[c.op_type for c in children]}", rm1)
        sq = next((c for c in children if c.op_type == "mul" and self._src(c.x) is d and self._src(c.y) is d), None)
        if sq is None:
            return self._fail(f"carré {[c.op_type for c in children]}", rm1)
        rm2 = self._single_child(sq.outputs[0], "reduce_mean")
        if rm2 is None:
            return self._fail("rm2", rm1)
        add = self._single_child(rm2.outputs[0], "add")
        if add is None:
            return self._fail(f"add {[c.op_type for c in rm2.outputs[0].child_ops]}", rm1)
        eps = _const_val(add.y) if add.x is rm2.outputs[0] else _const_val(add.x)
        if eps is None:
            return False
        sqrt = self._single_child(add.outputs[0], "sqrt")
        if sqrt is None:
            return self._fail(f"sqrt {[c.op_type for c in add.outputs[0].child_ops]}", rm1)
        div = self._single_child(sqrt.outputs[0], "real_div")
        if div is None or self._src(div.y) is not sqrt.outputs[0]:
            return self._fail(f"div {[c.op_type for c in sqrt.outputs[0].child_ops]}", rm1)
        one = _const_val(div.x)
        if one is None or not np.allclose(one, 1.0):
            return self._fail(f"one={one}", rm1)
        final = self._single_child(div.outputs[0], "mul")
        fx, fy = (self._src(final.x), self._src(final.y)) if final is not None else (None, None)
        if final is None or not ((fx is d and fy is div.outputs[0]) or (fy is d and fx is div.outputs[0])):
            return self._fail(f"final {[c.op_type for c in div.outputs[0].child_ops]}", rm1)
        C = x.shape[1]
        out_var = final.outputs[0]
        with block:
            new = mb.instance_norm(
                x=x, gamma=np.ones(C, dtype=np.float32), beta=np.zeros(C, dtype=np.float32),
                epsilon=float(np.array(eps).reshape(-1)[0]), before_op=rm1, name=out_var.name + "_in")
        block.replace_uses_of_var_after_op(anchor_op=final, old_var=out_var, new_var=new)
        block.remove_ops([rm1, sub, sq, rm2, add, sqrt, div, final] + list(dict.fromkeys(self._casts)))
        return True


print("== chargement ONNX", src)
model = onnx.load(src)
used = {i for n in model.graph.node for i in n.input}
keep = [t for t in model.graph.initializer if t.name in used]
del model.graph.initializer[:]; model.graph.initializer.extend(keep)
tm = onnx2torch.convert(model).eval()
traced = torch.jit.trace(tm, (torch.rand(1, 3, 128, 128), torch.randn(1, 512)))

pipeline = ct.PassPipeline.DEFAULT
if FUSE:
    names = pipeline.passes
    anchor = next((i for i, n in enumerate(names) if "cast_optimization" in n), 0)
    pipeline.insert_pass(anchor + 1, "custom::fuse_adain_instance_norm")
    print("passe insérée après", names[anchor] if anchor else "(début)")
print("== conversion (fusion instance_norm :", FUSE, ")")
t0 = time.time()
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="target", shape=(1, 3, 128, 128), dtype=np.float32),
            ct.TensorType(name="source", shape=(1, 512), dtype=np.float32)],
    outputs=[ct.TensorType(name="output", dtype=np.float32)],
    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE,
    pass_pipeline=pipeline,
)
print(f"conversion en {time.time()-t0:.0f}s")
from collections import Counter
c = Counter(op.op_type for op in mlmodel._mil_program.functions["main"].operations if op.op_type != "const")
print("ops :", dict(c.most_common()))
mlmodel.save(dst); print("sauvegardé :", dst)

print("== fidélité vs ONNX Runtime (CPU fp32)")
sess = ort.InferenceSession(src, providers=["CPUExecutionProvider"]); names = [i.name for i in sess.get_inputs()]
rng = np.random.default_rng(0)
tgt = rng.random((1, 3, 128, 128), dtype=np.float32)
lat = rng.standard_normal((1, 512)).astype(np.float32); lat /= np.linalg.norm(lat)
ref = sess.run(None, {names[0]: tgt, names[1]: lat})[0]
for cu in ("CPU_AND_NE", "ALL"):
    m = ct.models.MLModel(dst, compute_units=getattr(ct.ComputeUnit, cu))
    pred = m.predict({"target": tgt, "source": lat})["output"]
    diff = np.abs(pred - ref)
    for _ in range(5): m.predict({"target": tgt, "source": lat})
    l = []
    for _ in range(30):
        t = time.perf_counter(); m.predict({"target": tgt, "source": lat}); l.append(time.perf_counter() - t)
    print(f"{cu:11s}: écart max {diff.max():.4f} moyen {diff.mean():.5f} | latence méd {np.median(l)*1000:.0f} ms")
