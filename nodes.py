"""shotdrift nodes for ComfyUI: measure the camera move in the clip you just made.

Generative video is given a camera move and is under no obligation to deliver it.
The usual check is a person watching sixty clips and forming an impression - and
by the time they do, the graph that produced them has moved on. Measuring inside
the graph means the clip is judged where it was made, by the same run, before
anyone has to look at it.

Two nodes, because there are two honestly different things to measure:

  Shotdrift Measure        the IMAGE batch, in memory, before encoding. This is
                           the generator's actual output.
  Shotdrift Measure File   an encoded video on disk. Not the same question: that
                           file has been through a codec, and it is the thing
                           that ships.

The measurement itself lives in the `shotdrift` package, not here. A node that
reimplements the measurement is a second opinion that will drift from the first.
"""

from __future__ import annotations

from shotdrift import __version__ as SHOTDRIFT_VERSION
from shotdrift import measure, measure_frames, report
from shotdrift.expect import known as known_moves
from shotdrift.verdict import BROKEN, SOFT, at_or_above

NONE = "(none)"

# What a graph is allowed to stop for. Stopping is the point of gating a batch -
# an unattended run of 60 takes is only worth doing if the bad ones announce
# themselves - but the default stops for nothing, because a node that halts a
# queue the first time it is wired in gets deleted.
FAIL_MODES = [
    "nothing",
    "the declared move was not held",
    "a broken finding",
    "any finding",
]


def _should_fail(r, mode: str) -> str | None:
    """The reason to stop, or None. One place, so both nodes agree."""
    if mode == FAIL_MODES[1]:
        bad = [s for s in r.shots if s.expect is not None and not s.expect.ok]
        if bad:
            return f"declared move not held: {bad[0].expect.detail}"
    elif mode == FAIL_MODES[2]:
        f = at_or_above(r.findings, BROKEN)
        if f:
            return f"broken: {f[0].summary} ({f[0].evidence})"
    elif mode == FAIL_MODES[3]:
        f = at_or_above(r.findings, SOFT)
        if f:
            return f"{f[0].severity}: {f[0].summary} ({f[0].evidence})"
        bad = [s for s in r.shots if s.expect is not None and not s.expect.ok]
        if bad:
            return f"declared move not held: {bad[0].expect.detail}"
    return None


def _finish(r, fail_on: str):
    # Which version measured it. A number in a report with no provenance is the
    # thing you cannot reproduce six weeks later.
    text = f"shotdrift {SHOTDRIFT_VERSION}" + report(r)
    print(text)                       # the ComfyUI console is where people look
    reason = _should_fail(r, fail_on)
    if reason:
        # Raising is what stops the queue. The report goes with it, or the person
        # reading the error has to go and find out what happened separately.
        raise RuntimeError(f"shotdrift stopped this run - {reason}\n{text}")
    import json
    return text, bool(r.ok), json.dumps(r.as_dict(), indent=2)


_COMMON = {
    "expect": ([NONE] + known_moves(), {
        "tooltip": "The move you asked the generator for. Checked three ways: "
                   "did it happen, was it the right way round, was it held.",
    }),
    "fail_on": (FAIL_MODES, {
        "tooltip": "Stop the queue when this is true. Use it to gate an "
                   "unattended batch; leave it at 'nothing' to only report.",
    }),
    "max_side": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                         "tooltip": "Analysis resolution, long edge. Everything "
                                    "is reported in fractions of the frame "
                                    "width, so this does not move the numbers."}),
    "grid": ("INT", {"default": 4, "min": 2, "max": 8,
                     "tooltip": "Tiles per axis that get tracked and fitted."}),
    "segment": ("BOOLEAN", {"default": True,
                            "tooltip": "Detect cuts and measure each shot on "
                                       "its own. Off measures the batch as one "
                                       "take, which is wrong for an edit."}),
}


class ShotdriftMeasure:
    """Measure the camera move in an IMAGE batch, and pass the batch through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The frames to measure. At "
                                                "least two; motion is measured "
                                                "BETWEEN frames."}),
                **_COMMON,
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("images", "report", "held", "json")
    OUTPUT_TOOLTIPS = (
        "The frames, untouched - wire this through so the node sits inline.",
        "The human-readable measurement.",
        "True only if every shot is clean AND any declared move was held.",
        "The same measurement, machine-readable.",
    )
    FUNCTION = "run"
    CATEGORY = "shotdrift"
    DESCRIPTION = ("Measure whether a generated clip holds the camera move it was "
                   "given. Camera path from the pixels alone - pan, zoom, roll.")

    def run(self, images, expect, fail_on, max_side, grid, segment):
        r = measure_frames(
            images,
            expect=None if expect == NONE else expect,
            name=f"IMAGE batch ({int(images.shape[0])} frames)",
            max_side=max_side, grid=grid, segment=segment,
        )
        text, held, js = _finish(r, fail_on)
        # `images` is returned as it came in, not a copy: this node measures, it
        # does not process, and a node that quietly re-encodes what it was given
        # would make everything after it a measurement of the node.
        return (images, text, held, js)


class ShotdriftMeasureFile:
    """Measure an encoded video on disk - the file that actually ships."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False,
                                    "tooltip": "Path to a video file. Needs "
                                               "ffmpeg on PATH."}),
                **_COMMON,
            },
            "optional": {
                "max_frames": ("INT", {"default": 600, "min": 2, "max": 100000,
                                       "tooltip": "Frame budget. The report says "
                                                  "when a clip was truncated."}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("report", "held", "json")
    FUNCTION = "run"
    CATEGORY = "shotdrift"
    DESCRIPTION = ("Measure the camera move in a video file. Use after a save "
                   "node to measure what was actually written, codec included.")

    def run(self, path, expect, fail_on, max_side, grid, segment, max_frames=600):
        if not str(path).strip():
            raise ValueError("shotdrift: give a path to a video file.")
        r = measure(
            str(path).strip(),
            expect=None if expect == NONE else expect,
            max_side=max_side, grid=grid, segment=segment, max_frames=max_frames,
        )
        return _finish(r, fail_on)


NODE_CLASS_MAPPINGS = {
    "ShotdriftMeasure": ShotdriftMeasure,
    "ShotdriftMeasureFile": ShotdriftMeasureFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShotdriftMeasure": "Shotdrift Measure (frames)",
    "ShotdriftMeasureFile": "Shotdrift Measure (file)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "SHOTDRIFT_VERSION", "FAIL_MODES"]
