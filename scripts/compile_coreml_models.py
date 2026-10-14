"""Précompile les .mlpackage de models/ en .mlmodelc dans models/compiled/ (livrés avec l'app)."""
import os, glob, shutil, sys
import coremltools as ct
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "models", "compiled"); os.makedirs(out, exist_ok=True)
for pkg in sorted(glob.glob(os.path.join(root, "models", "*.mlpackage"))):
    name = os.path.splitext(os.path.basename(pkg))[0]
    dst = os.path.join(out, name + ".mlmodelc")
    w = os.path.join(pkg, "Data", "com.apple.CoreML", "weights", "weight.bin")
    if os.path.isdir(dst) and os.path.exists(w) and os.path.getmtime(dst) >= os.path.getmtime(w):
        print("à jour :", dst); continue
    tmp = ct.utils.compile_model(pkg)
    if os.path.isdir(dst): shutil.rmtree(dst)
    shutil.copytree(tmp, dst); print("compilé :", dst)
