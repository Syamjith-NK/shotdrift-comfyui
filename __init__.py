"""ComfyUI entry point. ComfyUI imports this directory and reads the mappings.

Loaded two ways on purpose:

  as a PACKAGE  - what ComfyUI does, and the relative import is correct there.
  as a FILE     - what anything walking the tree does, including pytest, which
                  imports every `__init__.py` between its rootdir and a test
                  file. With no parent package a relative import cannot resolve,
                  and the fallback loads the same module by its own path.

The fallback is narrow by design: it triggers only on the missing-parent case and
only for this one sibling file. A blanket `except ImportError` here would turn a
genuine packaging mistake into nodes that silently never appear.

A missing dependency is re-raised with the command that fixes it, and is NOT
swallowed - exporting empty mappings would leave ComfyUI starting cleanly with
the nodes absent, which is worse than a loud failure because nothing tells you
to go and look.
"""

_MISSING_SHOTDRIFT = (
    "shotdrift-comfyui needs the `shotdrift` package, which is not installed in "
    "ComfyUI's python. Install it into the SAME interpreter ComfyUI runs on:\n"
    "    path/to/comfy/python -m pip install "
    "git+https://github.com/Syamjith-NK/shotdrift\n"
    "It pulls in numpy and pillow only, and needs ffmpeg on PATH for the file node."
)


def _load_nodes():
    try:
        from . import nodes
        return nodes
    except ImportError as e:
        if isinstance(e, ModuleNotFoundError) and \
                (e.name or "").split(".")[0] == "shotdrift":
            raise ModuleNotFoundError(_MISSING_SHOTDRIFT) from e
        if "no known parent package" not in str(e):
            raise
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodes.py")
    spec = importlib.util.spec_from_file_location("shotdrift_comfyui_nodes", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "shotdrift":
            raise ModuleNotFoundError(_MISSING_SHOTDRIFT) from e
        raise
    return mod


_nodes = _load_nodes()
NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
