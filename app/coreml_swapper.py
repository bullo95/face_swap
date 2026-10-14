"""Backend CoreML natif pour inswapper (mlpackage produit par scripts/convert_inswapper_coreml.py).

Expose la même surface que la session ONNX Runtime utilisée par FaceEngine._swap_face :
`run(output_names, feed)` -> [array (1,3,128,128)], plus `get_inputs()`.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

COMPUTE_UNIT_MAP = {  # même vocabulaire que l'option CoreML d'ONNX Runtime
    "ALL": "ALL",
    "CPUAndNeuralEngine": "CPU_AND_NE",
    "CPUAndGPU": "CPU_AND_GPU",
    "CPUOnly": "CPU_ONLY",
}


class _Input:
    def __init__(self, name: str):
        self.name = name


class CoreMLSwapperSession:
    def __init__(self, mlpackage_path: str, compute_units: str = "CPUAndNeuralEngine"):
        import coremltools as ct

        if not os.path.isdir(mlpackage_path):
            raise FileNotFoundError(mlpackage_path)
        cu = getattr(ct.ComputeUnit, COMPUTE_UNIT_MAP.get(compute_units, "CPU_AND_NE"))
        self.model = ct.models.MLModel(mlpackage_path, compute_units=cu)
        spec = self.model.get_spec()
        self._in_names = [i.name for i in spec.description.input]
        self._out_name = spec.description.output[0].name
        self.compute_units = compute_units
        log.info("swapper CoreML natif : %s (%s)", os.path.basename(mlpackage_path), cu)

    def get_inputs(self):
        return [_Input(n) for n in self._in_names]

    def get_providers(self):
        return [f"CoreML natif ({self.compute_units})"]

    def run(self, output_names, feed: dict):
        # feed arrive avec les noms ONNX (target/source) ; on mappe par position si besoin
        if all(k in self._in_names for k in feed):
            inp = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feed.items()}
        else:
            vals = list(feed.values())
            inp = {n: np.ascontiguousarray(v, dtype=np.float32) for n, v in zip(self._in_names, vals)}
        out = self.model.predict(inp)[self._out_name]
        return [np.asarray(out, dtype=np.float32)]


class CoreMLSwapper:
    """Adaptateur minimal reproduisant les attributs de insightface INSwapper utilisés par le moteur."""

    def __init__(self, mlpackage_path: str, onnx_path: str, compute_units: str = "CPUAndNeuralEngine"):
        self.session = CoreMLSwapperSession(mlpackage_path, compute_units)
        # projection du vecteur d'identité (emap) : fichier .npy à côté du modèle, sinon lue dans l'ONNX
        npy = os.path.join(os.path.dirname(mlpackage_path), "inswapper_emap.npy")
        if os.path.exists(npy):
            self.emap = np.load(npy).astype(np.float32)
        else:
            import onnx
            from onnx import numpy_helper
            graph = onnx.load(onnx_path, load_external_data=False).graph
            self.emap = numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)
        self.input_mean = 0.0
        self.input_std = 255.0
        self.input_size = (128, 128)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = ["output"]
