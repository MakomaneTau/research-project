"""Verify compiled extraction, data access and CUDA before full experiments."""
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import pylidc as pl
import radiomics
import radiomics._cmatrices
import radiomics._cshape
import torch
from lidc_experiment.compat import load_pylidc

from lidc_experiment.features import build_extractor, extract_case
from lidc_experiment.training import CompactCNN
from lidc_experiment.utils import environment_record, write_json


def main():
    if not radiomics.__version__.lstrip("v").startswith("3.1.0"):
        raise RuntimeError(f"Unexpected PyRadiomics build {radiomics.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the primary workstation configuration")
    assert load_pylidc().query(pl.Scan).count() > 0
    extractor, settings = build_extractor({})
    yy, xx = np.ogrid[:64, :64]
    mask = ((xx - 32) ** 2 + (yy - 32) ** 2 <= 8 ** 2).astype(np.uint8)
    hu = (-800 + xx * 3 + yy * 2).astype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "known_slice.npz"
        np.savez(path, hu=hu, mask=mask)
        features, _ = extract_case(path, extractor)
        if not features or not np.isfinite(list(features.values())).all():
            raise RuntimeError("Synthetic 2D extraction failed")
        np.testing.assert_allclose(features["original_shape2D_PixelSurface"], float(mask.sum()), rtol=0, atol=1e-6)
        np.testing.assert_allclose(features["original_firstorder_Mean"], float(hu[mask > 0].mean()), rtol=0, atol=1e-5)
    network = CompactCNN(.2).cuda()
    inputs = torch.zeros((32, 2, 64, 64), device="cuda")
    inputs[:, 1, 24:40, 24:40] = 1
    inputs[:, 0] = inputs[:, 1] * .5
    loss = torch.nn.functional.binary_cross_entropy_with_logits(network(inputs), torch.zeros(32, device="cuda"))
    loss.backward()
    torch.cuda.synchronize()
    result = {"passed": True, "pyradiomics": radiomics.__version__, "features": len(features),
              "cuda_batch_size": 32, "cuda_loss": float(loss.detach().cpu()), "environment": environment_record()}
    write_json("artifacts/environment_smoke.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
