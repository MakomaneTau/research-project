"""Narrow compatibility for pylidc 0.2.3's removed NumPy scalar aliases.

Only pylidc module globals receive the adapter. NumPy itself and every other
package are unchanged; each removed alias has its original builtin meaning.
"""
import sys

import numpy as np


class _LegacyNumpy:
    int = int
    float = float
    bool = bool
    complex = complex
    object = object

    def __getattr__(self, name):
        return getattr(np, name)


def load_pylidc():
    import pylidc
    for name, module in list(sys.modules.items()):
        if name.startswith("pylidc") and module is not None and getattr(module, "np", None) is np:
            module.np = _LegacyNumpy()
    return pylidc
