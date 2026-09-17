import importlib
import os
import sys

_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

_nodes = importlib.import_module("sol_h3.nodes")
NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
