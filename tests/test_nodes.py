"""Tests for the ComfyUI nodes, driven with REAL torch tensors.

Two things are being checked and they are not the same thing:

  the CONTRACT - ComfyUI does not call these classes, it inspects them. It reads
                 INPUT_TYPES, RETURN_TYPES and FUNCTION and wires the graph from
                 what it finds, so a node can be perfectly correct and still not
                 appear, or appear with the wrong number of outputs. The contract
                 is therefore asserted, and asserted through the same import
                 mechanism ComfyUI uses - the package directory, not the module.

  the BOUNDARY - a torch tensor is not a numpy array. It is NHWC, it may be on a
                 device, and the conversion is the one place this package can be
                 wrong while `shotdrift` itself is right. So the fixtures are
                 real tensors, not duck types: a stub that answers `.cpu()` would
                 pass a test that the real thing fails.

What these do NOT cover, stated rather than implied: no ComfyUI is installed
here, so a real graph execution - queue, caching, the frontend rendering the
dropdown - is unverified.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load_like_comfyui():
    """Import the directory the way ComfyUI imports a custom node."""
    spec = importlib.util.spec_from_file_location(
        "shotdrift_comfyui_under_test", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


PKG = _load_like_comfyui()
NODES = PKG.NODE_CLASS_MAPPINGS


# ------------------------------------------------------------- fixtures ------
def _plate(w=720, h=720, seed=7):
    rng = np.random.default_rng(seed)
    low = rng.random((h // 10, w // 10)).astype(np.float32)
    img = np.asarray(
        Image.fromarray((low * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
        dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w]
    detail = 0.22 * np.sin(xx / 6.0) * np.cos(yy / 8.0)
    return np.clip(img * 0.72 + rng.random((h, w)).astype(np.float32) * 0.18 + detail,
                   0, 1).astype(np.float32)


def _crop(plate, cx, cy, size, out=320):
    h, w = plate.shape
    s = int(round(size))
    x0 = max(0, min(w - s, int(round(cx - s / 2))))
    y0 = max(0, min(h - s, int(round(cy - s / 2))))
    im = Image.fromarray((plate[y0:y0 + s, x0:x0 + s] * 255).astype(np.uint8))
    return np.asarray(im.resize((out, out), Image.BICUBIC), dtype=np.float32) / 255.0


def _batch(frames_gray) -> torch.Tensor:
    """Grey frames -> a ComfyUI IMAGE: torch float32 (B, H, W, 3) in 0..1."""
    a = np.stack(frames_gray)[..., None].repeat(3, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(a))


def push_in(n=36) -> torch.Tensor:
    p = _plate()
    t = np.linspace(0, 1, n)
    ease = t * t * (3 - 2 * t)
    return _batch([_crop(p, 360, 360, 600 - 230 * e) for e in ease])


def pan_right(n=36) -> torch.Tensor:
    p = _plate()
    t = np.linspace(0, 1, n)
    ease = t * t * (3 - 2 * t)
    return _batch([_crop(p, 220 + 280 * e, 360, 360) for e in ease])


# ------------------------------------------------------------- contract ------
def test_entry_point_loads_as_a_package_too():
    """ComfyUI imports this directory AS A PACKAGE. The module-level loader above
    covers the no-parent case; this covers the one that actually ships.
    """
    pkg_name = "shotdrift_comfyui_as_package"
    spec = importlib.util.spec_from_file_location(
        pkg_name, ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    assert set(mod.NODE_CLASS_MAPPINGS) == set(NODES)
    # ...and it used the real relative import, not the by-path fallback.
    assert f"{pkg_name}.nodes" in sys.modules


def test_nodes_module_exports_what_it_advertises():
    """A name in __all__ that does not exist breaks `import *` and nothing else
    notices - it happened here when an unused import was tidied away."""
    assert PKG.nodes.__all__, "nothing advertised: this test would pass vacuously"
    for name in PKG.nodes.__all__:
        assert hasattr(PKG.nodes, name), f"__all__ advertises missing {name}"


def test_both_nodes_are_registered_with_display_names():
    assert set(NODES) == {"ShotdriftMeasure", "ShotdriftMeasureFile"}
    assert set(PKG.NODE_DISPLAY_NAME_MAPPINGS) == set(NODES)


@pytest.mark.parametrize("key", ["ShotdriftMeasure", "ShotdriftMeasureFile"])
def test_node_contract(key):
    cls = NODES[key]
    spec = cls.INPUT_TYPES()
    assert "required" in spec and spec["required"]
    for name, decl in {**spec["required"], **spec.get("optional", {})}.items():
        assert isinstance(decl, tuple) and decl, f"{key}.{name} is not a declaration"
    assert isinstance(cls.RETURN_TYPES, tuple) and cls.RETURN_TYPES
    assert len(cls.RETURN_NAMES) == len(cls.RETURN_TYPES)
    assert callable(getattr(cls(), cls.FUNCTION))


def test_declared_moves_match_the_library():
    from shotdrift.expect import known
    choices = NODES["ShotdriftMeasure"].INPUT_TYPES()["required"]["expect"][0]
    assert choices[0] == "(none)"
    assert choices[1:] == known()


def test_function_signature_accepts_every_declared_input():
    """ComfyUI calls FUNCTION with the input names as keywords. A declared input
    the method cannot accept is a TypeError at queue time, not at load time."""
    import inspect
    for key, cls in NODES.items():
        spec = cls.INPUT_TYPES()
        declared = set(spec["required"]) | set(spec.get("optional", {}))
        params = set(inspect.signature(getattr(cls(), cls.FUNCTION)).parameters)
        assert declared <= params, f"{key} cannot accept {declared - params}"


# ------------------------------------------------------------- boundary ------
def test_measures_a_real_torch_batch_and_holds_the_move():
    images, text, held, js = NODES["ShotdriftMeasure"]().run(
        push_in(), "push-in", "nothing", 512, 4, True)
    assert held is True
    assert "HELD" in text and "NOT HELD" not in text
    import json
    assert json.loads(js)["shots"][0]["expect"]["ok"] is True


def test_a_pan_sold_as_a_push_in_does_not_hold():
    _, text, held, _ = NODES["ShotdriftMeasure"]().run(
        pan_right(), "push-in", "nothing", 512, 4, True)
    assert held is False
    assert "NOT HELD" in text


def test_images_pass_through_untouched():
    """The node measures; anything downstream must get the same object back."""
    batch = push_in()
    out = NODES["ShotdriftMeasure"]().run(batch, "(none)", "nothing", 512, 4, True)[0]
    assert out is batch
    assert torch.equal(out, batch)


def test_uint8_tensor_is_not_measured_as_if_it_were_zero_to_one():
    """A 0..255 batch read as 0..1 clips every frame to white and measures
    nothing. Same clip, both dtypes, same verdict."""
    f = push_in()
    a = NODES["ShotdriftMeasure"]().run(f, "push-in", "nothing", 512, 4, True)
    b = NODES["ShotdriftMeasure"]().run(
        (f * 255).round().to(torch.uint8), "push-in", "nothing", 512, 4, True)
    assert a[2] is True and b[2] is True


def test_channels_first_tensor_is_refused_by_name():
    with pytest.raises(ValueError, match="channels-first"):
        NODES["ShotdriftMeasure"]().run(
            push_in().permute(0, 3, 1, 2), "(none)", "nothing", 512, 4, True)


def test_single_frame_batch_is_refused_not_guessed():
    with pytest.raises(ValueError, match="at least two"):
        NODES["ShotdriftMeasure"]().run(
            push_in()[:1], "(none)", "nothing", 512, 4, True)


# ----------------------------------------------------------------- gate ------
def test_fail_on_declared_move_stops_the_run_and_says_why():
    with pytest.raises(RuntimeError) as e:
        NODES["ShotdriftMeasure"]().run(
            pan_right(), "push-in", "the declared move was not held", 512, 4, True)
    assert "declared move not held" in str(e.value)
    assert "camera path" in str(e.value), "the report must travel with the error"


def test_fail_on_nothing_never_stops_a_run():
    _, _, held, _ = NODES["ShotdriftMeasure"]().run(
        pan_right(), "push-in", "nothing", 512, 4, True)
    assert held is False          # reported, not raised


def test_a_held_move_does_not_trip_any_gate():
    for mode in PKG.nodes.FAIL_MODES:
        _, _, held, _ = NODES["ShotdriftMeasure"]().run(
            push_in(), "push-in", mode, 512, 4, True)
        assert held is True


def test_file_node_refuses_an_empty_path():
    with pytest.raises(ValueError, match="path to a video file"):
        NODES["ShotdriftMeasureFile"]().run("  ", "(none)", "nothing", 512, 4, True)


@pytest.mark.parametrize("mode", PKG.nodes.FAIL_MODES[1:])
def test_unmeasurable_batch_blocks_every_active_gate(mode):
    with pytest.raises(RuntimeError, match="incomplete or unmeasurable"):
        NODES["ShotdriftMeasure"]().run(
            torch.zeros((20, 128, 128, 3)), "static", mode, 512, 4, True)


def test_report_only_unknown_is_false_and_strict_json():
    import json
    _, text, held, js = NODES["ShotdriftMeasure"]().run(
        torch.zeros((20, 128, 128, 3)), "static", "nothing", 512, 4, True)
    def invalid_constant(value):
        raise ValueError(f"invalid JSON constant {value}")
    data = json.loads(js, parse_constant=invalid_constant)
    assert not held and data["verdict"] == "unknown"
    assert "NOT MEASURABLE" in text and "HELD" not in text


def test_file_node_is_a_terminal_output_and_never_reuses_external_files(tmp_path):
    cls = NODES["ShotdriftMeasureFile"]
    assert cls.OUTPUT_NODE
    path = tmp_path / "take.mp4"
    path.write_bytes(b"first")
    a = cls.IS_CHANGED(str(path), "static", "nothing", 512, 4, True)
    path.write_bytes(b"other")
    b = cls.IS_CHANGED(str(path), "static", "nothing", 512, 4, True)
    assert a != b
    # Also bypass cache before an upstream saver changes the existing file.
    assert cls.IS_CHANGED(str(path), "static", "nothing", 512, 4, True) != b


def test_incomplete_result_blocks_without_an_expectation():
    from shotdrift import Result
    for mode in PKG.nodes.FAIL_MODES[1:]:
        assert PKG.nodes._should_fail(Result(clip="empty"), mode)
