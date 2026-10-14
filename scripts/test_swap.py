"""Test hors caméra : swap du visage source sur une image cible, avec et sans GFPGAN."""
import logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)
import cv2
from app.engine import FaceEngine, Settings

src, dst, out = sys.argv[1], sys.argv[2], sys.argv[3]
eng = FaceEngine(Settings(many_faces=True))
t = time.perf_counter(); eng.load(progress=print); print(f"load: {time.perf_counter()-t:.1f}s")
assert eng.set_source_image(src), "pas de visage source"
img = cv2.imread(dst)
for enh in (False, True):
    eng.settings.enhance = enh
    eng.process(img)  # chauffe
    t = time.perf_counter(); n = 5
    for _ in range(n):
        res = eng.process(img)
    dt = (time.perf_counter() - t) / n
    print(f"enhance={enh}: {dt*1000:.0f} ms/frame ({1/dt:.1f} fps) sur {img.shape[1]}x{img.shape[0]}")
    cv2.imwrite(out.replace(".jpg", f"_enh{int(enh)}.jpg"), res)
print("ok")
