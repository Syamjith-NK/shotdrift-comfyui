"""Run a model-free smoke test with a real installed ComfyUI executor.

Use the ComfyUI interpreter with shotdrift >= 0.2.2 installed, and FFmpeg/FFprobe
on PATH. This registers the nodes only in this process, not in the desktop app.
"""

import argparse
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--output", help="optional JSON results file")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(args.comfy_root).resolve()))
    sys.argv = ["shotdrift-smoke", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import execution
    import folder_paths
    import nodes
    import numpy as np
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "shotdrift_smoke", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    nodes.NODE_CLASS_MAPPINGS.update(package.NODE_CLASS_MAPPINGS)

    class Server:
        client_id = None
        last_node_id = None

        def send_sync(self, *args):
            pass

    with tempfile.TemporaryDirectory(prefix="shotdrift-comfy-") as directory:
        directory = Path(directory)
        for name in ("input", "output", "temp"):
            target = directory / name
            target.mkdir()
            getattr(folder_paths, "set_" + name + "_directory")(str(target))

        rng = np.random.default_rng(7)
        low = Image.fromarray((rng.random((56, 56)) * 255).astype(np.uint8))
        plate = np.asarray(low.resize((560, 560), Image.Resampling.BICUBIC), dtype=np.float32)
        plate = np.clip(0.8 * plate + 0.2 * rng.random((560, 560)) * 255, 0, 255).astype(np.uint8)
        plate = Image.fromarray(plate)

        def video(path, kind):
            frames = []
            for t in np.linspace(0, 1, 36):
                e = t * t * (3 - 2 * t)
                size = round(460 - 170 * e) if kind == "push" else 280
                cx = 280 if kind == "push" else 170 + 220 * e
                left, top = round(cx - size / 2), round(280 - size / 2)
                crop = plate.crop((left, top, left + size, top + size))
                frames.append(np.asarray(crop.resize((256, 256), Image.Resampling.BICUBIC)))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
                            "-pix_fmt", "gray", "-s", "256x256", "-r", "24", "-i", "-",
                            "-c:v", "ffv1", str(path)], input=np.stack(frames).tobytes(), check=True)

        push, pan, current = (directory / name for name in ("push.mkv", "pan.mkv", "current.mkv"))
        video(push, "push")
        video(pan, "pan")
        shutil.copyfile(push, current)
        prompt = json.loads((root / "docs/workflows/file-check.api.json").read_text())
        prompt["1"]["inputs"].update(path=str(current), fail_on="nothing")
        validation = asyncio.run(execution.validate_prompt("standalone", copy.deepcopy(prompt), None))
        assert validation[0], validation
        executor = execution.PromptExecutor(
            Server(), cache_type=execution.CacheType.CLASSIC,
            cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})

        def execute(graph, label, expected):
            executor.execute(copy.deepcopy(graph), label, {}, ["1"])
            assert executor.success, executor.status_messages
            entry = asyncio.run(executor.caches.outputs.get("1"))
            held = entry.outputs[1][0]
            assert held is expected, (label, entry.outputs)

        execute(prompt, "initial-push", True)
        shutil.copyfile(pan, current)
        execute(prompt, "same-path-pan", False)
        failing = copy.deepcopy(prompt)
        failing["1"]["inputs"]["fail_on"] = "the declared move was not held"
        executor.execute(failing, "reject-pan", {}, ["1"])
        assert not executor.success
        execute(prompt, "next-prompt", False)

        # Simulate a saver that rewrites the same path within each graph run.
        state = {"source": push}

        class Saver:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            @classmethod
            def IS_CHANGED(cls):
                return float("nan")

            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"
            CATEGORY = "test"

            def run(self):
                shutil.copyfile(state["source"], current)
                return (str(current),)

        nodes.NODE_CLASS_MAPPINGS["ShotdriftTestSaver"] = Saver
        ordered = copy.deepcopy(prompt)
        ordered["0"] = {"class_type": "ShotdriftTestSaver", "inputs": {}}
        ordered["1"]["inputs"]["after"] = ["0", 0]
        execute(ordered, "save-then-push-check", True)
        state["source"] = pan
        execute(ordered, "save-then-pan-check", False)

    result = dict(standalone=True, changed_file_remeasured=True,
                  rejection=True, next_prompt_runs=True, upstream_save_order=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    print("COMFYUI INTEGRATION PASSED", json.dumps(result))


if __name__ == "__main__":
    main()
