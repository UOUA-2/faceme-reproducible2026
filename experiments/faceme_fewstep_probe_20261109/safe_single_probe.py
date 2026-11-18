"""Layered memory-safe single-image FaceMe probe.

Only this independent wrapper is changed.  It keeps 512x512 inputs, the
existing local weights, prompt, scheduler, seed, and ControlNet scale.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import run_probe as prior


PROJECT = prior.PROJECT
EXP = prior.EXP
SOURCE_INFER = prior.SOURCE_INFER
RUNNER_DIR = prior.RUNNER_DIR
DATASET_ROOT = prior.DATASET_ROOT
MODEL = prior.MODEL
CONTROLNET = prior.CONTROLNET
MIX = prior.MIX
PHOTO = prior.PHOTO
SEED = prior.SEED
SCALE = prior.SCALE
PROMPT = prior.PROMPT
NEG_PROMPT = prior.NEG_PROMPT
STEPS = prior.STEPS
SAFE_OUT = EXP / "safe_outputs"
LOG = EXP / "logs/safe_single_probe.log"
IDENTITY = ("LFW-Test-Ref", "Lloyd_Mudiwa_0001")


def log(message: str) -> None:
    line = time.strftime("[%Y-%m-%d %H:%M:%S]") + " " + message
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def mem(torch) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 3),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 3),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 3),
    }


def nvidia_snapshot() -> str:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20,
        )
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        return json.dumps({"gpu": p.stdout.strip(), "compute_apps": q.stdout.strip(), "stderr": p.stderr.strip()}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": repr(exc)}, ensure_ascii=False)


def exec_infer_namespace(torch, transformers):
    """Import infer.py definitions without running its main or changing it."""
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    ns = {"__name__": "safe_single_infer_runtime", "__file__": str(SOURCE_INFER)}
    exec(compile(SOURCE_INFER.read_text(encoding="utf-8"), str(SOURCE_INFER), "exec"), ns)
    cache: dict[str, object] = {}

    def patch_loader(cls, variant=None):
        original = cls.from_pretrained.__func__

        def loader(c, *args, **kwargs):
            kwargs.setdefault("low_cpu_mem_usage", True)
            kwargs.setdefault("torch_dtype", torch.float16)
            if variant:
                kwargs["variant"] = variant
            obj = original(c, *args, **kwargs)
            if kwargs.get("subfolder"):
                cache[kwargs["subfolder"]] = obj
            return obj

        cls.from_pretrained = classmethod(loader)

    patch_loader(ns["OriginalUNet2DConditionModel"], "fp16")
    patch_loader(ns["AutoencoderKL"])
    patch_loader(ns["ControlNetModel"])
    for name in ("CLIPTextModel", "CLIPTextModelWithProjection"):
        if hasattr(transformers, name):
            patch_loader(getattr(transformers, name))

    pipe_cls = ns["StableDiffusionXLControlNetPipeline"]
    original_pipe = pipe_cls.from_pretrained.__func__

    def pipe_loader(c, *args, **kwargs):
        if "text_encoder" not in kwargs and "text_encoder" in cache:
            kwargs["text_encoder"] = cache["text_encoder"]
        if "text_encoder_2" not in kwargs and "text_encoder_2" in cache:
            kwargs["text_encoder_2"] = cache["text_encoder_2"]
        return original_pipe(c, *args, **kwargs)

    pipe_cls.from_pretrained = classmethod(pipe_loader)
    ns["hf_hub_download"] = lambda *args, **kwargs: str(PHOTO)

    original_face = ns["FaceAnalysis2"]

    class CPUFaceAnalysis2(original_face):
        def __init__(self, *args, **kwargs):
            kwargs["providers"] = ["CPUExecutionProvider"]
            super().__init__(*args, **kwargs)

    ns["FaceAnalysis2"] = CPUFaceAnalysis2
    ns["_safe_cache"] = cache
    return ns


def module_info(name: str, module) -> dict[str, object]:
    params = list(module.parameters())
    if not params:
        return {"name": name, "parameters": 0}
    devices = sorted({str(p.device) for p in params})
    dtypes = sorted({str(p.dtype) for p in params})
    return {"name": name, "parameters": len(params), "devices": devices, "dtypes": dtypes,
            "first_device": str(params[0].device), "first_dtype": str(params[0].dtype),
            "has_accelerate_hook": bool(getattr(module, "_hf_hook", None))}


def load_state(level: str, torch, transformers):
    ns = exec_infer_namespace(torch, transformers)
    args = argparse.Namespace(
        pretrained_model_name_or_path=MODEL, controlnet_model_name_or_path=CONTROLNET,
        mix_path=MIX, seed=SEED, input_dir=None, result_dir=None, ref_dir=None,
        pos_prompt=PROMPT, neg_prompt=NEG_PROMPT, key_word="face", color_correction=True,
    )
    mask, pos, pos_pool, neg, neg_pool = ns["prepare_prompt"](args)
    scheduler = ns["DDPMScheduler"].from_pretrained(MODEL, subfolder="scheduler")
    vae = ns["AutoencoderKL"].from_pretrained(MODEL, subfolder="vae", revision=None)
    unet = ns["OriginalUNet2DConditionModel"].from_pretrained(MODEL, subfolder="unet")
    id_state, lora_state = ns["load_photomaker_adapter"](str(PHOTO.parent), subfolder="", weight_name=PHOTO.name)
    lora_cfg = ns["LoraConfig"](r=64, lora_alpha=64, init_lora_weights="gaussian",
                                  target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet = ns["get_peft_model"](unet, lora_cfg)
    lora_state = {k.replace("unet.", "base_model.model.").replace(".weight", ".default.weight"): v
                  for k, v in lora_state.items() if k.startswith("unet.")}
    lora_state = ns["convert_unet_state_dict_to_peft"](lora_state)
    for name, param in unet.named_parameters():
        if name in lora_state:
            param.data.copy_(lora_state[name])
    unet.merge_and_unload()
    ident = ns["PhotoMakerIDEncoder"]()
    ident.load_state_dict(id_state, strict=False)
    controlnet = ns["ControlNetModel"].from_pretrained(CONTROLNET, local_files_only=True)
    mix = ns["Mix"]()
    mix.from_pretrained(MIX)
    app = ns["FaceAnalysis2"](name="antelopev2", root="./", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))
    for obj in (vae, unet, mix, ident, controlnet):
        obj.requires_grad_(False)
        obj.eval()
    dtype = torch.float16
    if level == "direct":
        for obj in (vae, unet, mix, ident, controlnet):
            obj.to("cuda", dtype=dtype)
    else:
        mix.to("cuda", dtype=dtype)
        ident.to("cuda", dtype=dtype)
        # Pipeline CPU-offload modes move denoising/VAE modules on demand.
        for obj in (vae, unet, controlnet):
            obj.to("cpu", dtype=dtype)
    pipe = ns["StableDiffusionXLControlNetPipeline"].from_pretrained(
        MODEL, vae=vae, controlnet=controlnet, unet=unet, torch_dtype=dtype,
    )
    pipe.scheduler = scheduler
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    xformers = importlib.util.find_spec("xformers") is not None
    attention_backend = "pytorch_sdpa_default"
    if xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()
        attention_backend = "xformers"
    if level == "direct":
        pipe.to("cuda")
    elif level == "model_cpu_offload":
        pipe.enable_model_cpu_offload()
    elif level == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(level)
    pipe.set_progress_bar_config(disable=True)
    state = {"ns": ns, "torch": torch, "vae": vae, "app": app, "pipeline": pipe,
             "scheduler": scheduler, "mask": mask, "pos": pos, "pos_pool": pos_pool,
             "neg": neg, "neg_pool": neg_pool, "ident": ident, "mix": mix,
             "clip": ns["CLIPImageProcessor"](), "level": level,
             "attention_backend": attention_backend}
    state["diagnostics"] = [
        module_info("unet", pipe.unet), module_info("controlnet", pipe.controlnet),
        module_info("vae", pipe.vae), module_info("text_encoder", pipe.text_encoder),
        module_info("text_encoder_2", pipe.text_encoder_2), module_info("mix", mix),
        module_info("photomaker_id_encoder", ident),
    ]
    return state


def release(state, torch) -> None:
    try:
        if state:
            pipe = state.get("pipeline")
            if pipe is not None:
                pipe.remove_all_hooks()
            state.clear()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_single_identity(level: str, state, torch, row: dict[str, str]) -> dict[str, object]:
    ident = f"{row['dataset']}__{row['identity']}"
    ctx = prior.build_identity_context(None, state, row)
    noisy, prep_ms, control = prior.prepare_noisy(state, ctx, SEED)
    ctx["control"] = control
    paths: dict[int, Path] = {}
    records = []
    for steps in STEPS:
        log(f"{level}: Lloyd steps={steps}")
        path = SAFE_OUT / level / ident / f"step_{steps}.png"
        before = mem(torch)
        try:
            image, infer_ms, peak = prior.call_single(state, ctx, noisy, steps)
            post_ms, arr = prior.postprocess(state, ctx, image, path)
        except Exception as exc:
            raise RuntimeError(f"{level} failed at steps={steps}: {exc}") from exc
        after = mem(torch)
        paths[steps] = path
        fm = prior.face_metrics(state, ctx, arr)
        finite = bool(np.isfinite(arr).all())
        black = bool(prior.anomaly(arr))
        rec = {"identity": ident, "lq_path": str(ctx["group"] / "lq/LQ.png"),
               "reference_path": str(ctx["refs"][0]), "steps": steps, "seed": SEED,
               "mode": level, "inference_time_ms": infer_ms, "preprocessing_time_ms": prep_ms,
               "identity_feature_extraction_time_ms": ctx["identity_ms"], "postprocessing_time_ms": post_ms,
               "total_time_ms": round(prep_ms + ctx["identity_ms"] + infer_ms + post_ms, 3),
               "peak_vram_mb": peak, "finite": finite, "black_or_white_anomaly": black,
               "output_path": str(path), "memory_before": before, "memory_after": after, **fm}
        records.append(rec)
        if not finite or black or not fm.get("face_detected"):
            raise RuntimeError(f"invalid output at steps={steps}: finite={finite}, anomaly={black}, face={fm.get('face_detected')}")
        del image, arr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    sheet = SAFE_OUT / level / ident / "contact_sheet_LQ_Reference_1_2_4_6_8_10_50.png"
    prior.make_sweep_sheet(ctx, paths, sheet)
    return {"level": level, "identity": ident, "records": records, "sheet": str(sheet),
            "identity_ms": ctx["identity_ms"], "preprocessing_ms": prep_ms}


def compare_history(state, current: dict[str, object], row: dict[str, str]) -> dict[str, object]:
    current_path = Path(current["records"][5]["output_path"])  # steps [1,2,4,6,8,10,50]
    history = PROJECT / "outputs/steps_compare_lloyd_10_v1/step_10/LQ_512.png"
    result = {"historical_path": str(history), "historical_exists": history.exists(),
              "current_path": str(current_path), "comparable": False}
    if not history.exists():
        return result
    a = cv2.imread(str(current_path), cv2.IMREAD_COLOR)
    b = cv2.imread(str(history), cv2.IMREAD_COLOR)
    if a is None or b is None or a.shape != b.shape:
        result["error"] = "decode or shape mismatch"
        return result
    diff = a.astype(np.float32) - b.astype(np.float32)
    result.update({"comparable": True, "historical_sha256": sha256(history),
                   "mean_abs_pixel_diff": float(np.mean(np.abs(diff))),
                   "rmse": float(np.sqrt(np.mean(diff * diff))),
                   "max_abs_pixel_diff": int(np.max(np.abs(diff)))})
    # This is only a severe-regression gate, not a claim of pixel equality.
    result["severe_difference_gate_passed"] = bool(result["mean_abs_pixel_diff"] <= 25.0 and result["rmse"] <= 40.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["direct", "model_cpu_offload", "sequential_cpu_offload"], default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    os.chdir(PROJECT)
    row = next(r for r in prior.read_selected_rows() if (r["dataset"], r["identity"]) == IDENTITY)
    import torch
    import transformers
    diagnostic = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "nvidia_before": nvidia_snapshot(),
                  "torch_before": mem(torch), "levels": [], "xformers_installed": importlib.util.find_spec("xformers") is not None,
                  "sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention")}
    write_json(EXP / "safe_probe_preflight.json", diagnostic)
    selected_level = None
    selected_result = None
    levels = [args.level] if args.level else ["direct", "model_cpu_offload", "sequential_cpu_offload"]
    for level in levels:
        log(f"starting level={level}")
        state = None
        entry: dict[str, object] = {"level": level, "nvidia_before": nvidia_snapshot(), "torch_before": mem(torch)}
        try:
            state = load_state(level, torch, transformers)
            entry["module_diagnostics"] = state["diagnostics"]
            entry["attention_backend"] = state["attention_backend"]
            entry["torch_after_load"] = mem(torch)
            result = run_single_identity(level, state, torch, row)
            entry["status"] = "passed"
            entry["result"] = result
            entry["nvidia_after"] = nvidia_snapshot()
            entry["torch_after"] = mem(torch)
            diagnostic["levels"].append(entry)
            selected_level = level
            selected_result = result
            write_json(EXP / "safe_probe_diagnostics.json", diagnostic)
            log(f"level={level} passed all single-identity steps")
            break
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()
            entry["nvidia_after"] = nvidia_snapshot()
            entry["torch_after"] = mem(torch)
            diagnostic["levels"].append(entry)
            write_json(EXP / "safe_probe_diagnostics.json", diagnostic)
            log(f"level={level} failed: {exc}")
            release(state, torch)
            state = None
    if selected_level is None:
        diagnostic["status"] = "stopped_single_image_oom_or_failure"
        diagnostic["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        diagnostic["nvidia_after"] = nvidia_snapshot()
        write_json(EXP / "safe_probe_diagnostics.json", diagnostic)
        return 2

    history = compare_history(state, selected_result, row)
    write_json(EXP / "historical_10_comparison.json", history)
    if history.get("comparable") and not history.get("severe_difference_gate_passed"):
        diagnostic["status"] = "stopped_history_difference"
        diagnostic["historical_comparison"] = history
        write_json(EXP / "safe_probe_diagnostics.json", diagnostic)
        release(state, torch)
        return 3

    # Only after Lloyd completes all seven steps and the historical 10-step
    # severe-regression gate passes do we expand to the original five train IDs.
    all_rows = prior.read_selected_rows()
    all_results = [selected_result]
    for r in all_rows:
        if (r["dataset"], r["identity"]) == IDENTITY:
            continue
        log(f"expanding to {r['dataset']}::{r['identity']} at level={selected_level}")
        all_results.append(run_single_identity(selected_level, state, torch, r))
    csv_rows = [row for result in all_results for row in result["records"]]
    fields = list(csv_rows[0])
    with (EXP / "step_sweep_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(csv_rows)
    diagnostic.update({"status": "completed", "selected_level": selected_level,
                       "historical_comparison": history, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                       "nvidia_after": nvidia_snapshot(), "torch_after": mem(torch)})
    write_json(EXP / "safe_probe_diagnostics.json", diagnostic)
    release(state, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
