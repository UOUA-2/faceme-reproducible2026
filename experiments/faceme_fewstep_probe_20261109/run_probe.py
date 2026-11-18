"""Strict, isolated FaceMe few-step and batch feasibility probe.

This wrapper reuses the existing adaptive_budget_v1 runtime loader without
editing formal FaceMe source or any historical output.  All writes go under
the experiment directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path("/mnt/f/ICVIP/DaChuang")
EXP = PROJECT / "experiments/faceme_fewstep_probe_20261109"
OUT = EXP / "outputs"
LOG = EXP / "logs/run.log"
FROZEN = PROJECT / "experiment_archive/faceme_controller_100_20261015T014246Z/analysis/controller_training_100_v1/FROZEN_SPLIT_MANIFEST.csv"
FINAL20 = PROJECT / "sample_input/controller_final_test_20_v1/manifest.csv"
SOURCE_INFER = PROJECT / "infer.py"
RUNNER_DIR = PROJECT / "experiments/adaptive_budget_v1"
DATASET_ROOT = PROJECT / "sample_input/controller_dataset_100"
MODEL = "models/SG161222--RealVisXL_V3.0"
CONTROLNET = "weights/controlnet"
MIX = "weights/mix"
PHOTO = PROJECT / "models/TencentARC--PhotoMaker/photomaker-v1.bin"
SEED = 42
STEPS = [1, 2, 4, 6, 8, 10, 50]
SCALE = 1.0
PROMPT = "A photo of face."
NEG_PROMPT = "A photo of face."
SELECTED = [
    ("LFW-Test-Ref", "Lloyd_Mudiwa_0001"),
    ("LFW-Test-Ref", "Caroline_Link_0001"),
    ("WebPhoto-Test-Ref", "00002_00"),
    ("WebPhoto-Test-Ref", "10025_00"),
    ("Wider-Test-Ref", "0481"),
]


class StopExperiment(RuntimeError):
    pass


def log(msg: str) -> None:
    line = time.strftime("[%Y-%m-%d %H:%M:%S]") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def read_selected_rows() -> list[dict[str, str]]:
    with FROZEN.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {(r["dataset"], r["identity"]): r for r in rows}
    final_keys = set()
    if FINAL20.exists():
        with FINAL20.open(encoding="utf-8-sig", newline="") as f:
            final_keys = {(r["dataset"], r["identity"]) for r in csv.DictReader(f)}
    selected = []
    for key in SELECTED:
        if key not in lookup:
            raise StopExperiment(f"selected identity missing from frozen split: {key}")
        row = lookup[key]
        if row.get("split") != "train":
            raise StopExperiment(f"selected identity is not train/development split: {key} -> {row.get('split')}")
        if key in final_keys:
            raise StopExperiment(f"selected identity overlaps final20: {key}")
        selected.append(row)
    for row in selected:
        group = DATASET_ROOT / row["dataset"] / row["identity"]
        paths = [group / "lq/LQ.png"] + [group / f"ref/{i}.png" for i in range(3)]
        if not all(p.is_file() for p in paths):
            raise StopExperiment(f"incomplete selected input: {row['dataset']}::{row['identity']}")
        for p in paths:
            with Image.open(p) as im:
                im.load()
                if (im.width, im.height, im.mode) != (512, 512, "RGB"):
                    raise StopExperiment(f"input is not 512x512 RGB: {p}")
    return selected


def input_records(row: dict[str, str]) -> dict[str, object]:
    group = DATASET_ROOT / row["dataset"] / row["identity"]
    paths = [group / "lq/LQ.png"] + [group / f"ref/{i}.png" for i in range(3)]
    return {"lq": str(paths[0]), "refs": [str(p) for p in paths[1:]],
            "sha256": {str(p): sha256(p) for p in paths}}


def load_runtime():
    """Load the already-used project runtime loader with redirected constants."""
    sys.path.insert(0, str(RUNNER_DIR))
    import run_teacher_outputs as base  # type: ignore

    base.PROJECT = PROJECT
    base.DATASET_ROOT = DATASET_ROOT
    base.OUTPUT_ROOT = OUT
    base.SOURCE_INFER = SOURCE_INFER
    base.PHOTO = PHOTO
    base.MODEL = MODEL
    base.CONTROLNET = CONTROLNET
    base.MIX = MIX
    base.SEED = SEED
    base.SCALE = SCALE
    base.PROMPT = PROMPT
    base.NEG_PROMPT = NEG_PROMPT
    os.chdir(PROJECT)
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    import torch
    import transformers

    ns = base.runtime_namespace(SOURCE_INFER.read_text(encoding="utf-8"), torch, transformers)
    state = base.load_state(ns, torch)
    return base, state


def build_identity_context(base, state, row: dict[str, str]) -> dict[str, object]:
    """Build the cached identity conditioning and measure it separately."""
    torch = state["torch"]
    ns = state["ns"]
    group = DATASET_ROOT / row["dataset"] / row["identity"]
    refs = [group / f"ref/{i}.png" for i in range(3)]
    t0 = time.perf_counter()
    ids, clips, ref_numpy = [], [], []
    for path in refs:
        ref = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        face = ns["analyze_faces"](state["app"], ref)[0]
        emb = torch.tensor(face["embedding"], device="cuda")
        emb = emb / torch.norm(emb, dim=0, keepdim=True)
        ids.append(emb.unsqueeze(0).to(dtype=torch.float16))
        crop = Image.fromarray(ref).crop(face["bbox"])
        crop = torch.tensor(state["clip"](crop)["pixel_values"][0], device="cuda").unsqueeze(0)
        clips.append(state["ident"](crop).to(dtype=torch.float16))
        vec = face["embedding"].astype(np.float32)
        vec /= max(np.linalg.norm(vec), 1e-12)
        ref_numpy.append(vec)
    emb_mix = state["mix"](clip_emb=torch.cat(clips), id_emb=torch.cat(ids))
    prompt = torch.cat([
        state["pos"][:, :state["mask"]].clone().to("cuda"),
        emb_mix.unsqueeze(0),
        state["pos"][:, state["mask"] + 1:].clone().to("cuda"),
    ], dim=1)[:, :77]
    identity_ms = ms(time.perf_counter() - t0)
    return {"row": row, "group": group, "refs": refs, "prompt": prompt,
            "ref_embeddings": ref_numpy, "identity_ms": identity_ms}


def prepare_noisy(state, ctx: dict[str, object], seed: int) -> tuple[object, float, object]:
    """Create one deterministic noisy latent; returns latent, preprocessing ms, control."""
    torch = state["torch"]
    group = ctx["group"]
    t0 = time.perf_counter()
    lq = group / "lq/LQ.png"
    control_np = cv2.cvtColor(cv2.imread(str(lq)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    control = torch.tensor(control_np * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    base_latent = state["vae"].encode(control).latent_dist.sample() * state["vae"].config.scaling_factor
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    noise = torch.randn_like(base_latent)
    ts = torch.randint(state["scheduler"].config.num_train_timesteps - 1,
                       state["scheduler"].config.num_train_timesteps,
                       (base_latent.shape[0],), device=base_latent.device).long()
    noisy = state["scheduler"].add_noise(base_latent, noise, ts)
    return noisy, ms(time.perf_counter() - t0), control


def postprocess(state, ctx: dict[str, object], image, path: Path) -> tuple[float, np.ndarray]:
    torch = state["torch"]
    t0 = time.perf_counter()
    control = ctx["control"]
    sr = torch.tensor((np.array(image) / 255.0) * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)
    sr = state["ns"]["wavelet_reconstruction"](sr.to(torch.float32).cpu(), control.to(torch.float32).cpu())
    arr = sr.squeeze().permute(1, 2, 0).mul(127.5).add(127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return ms(time.perf_counter() - t0), arr


def anomaly(arr: np.ndarray) -> bool:
    return (not np.isfinite(arr).all()) or (arr.std() < 1.0 and (arr.mean() < 2.0 or arr.mean() > 253.0))


def call_single(state, ctx, noisy, steps: int):
    torch = state["torch"]
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sync(torch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        image = state["pipeline"](
            latents=noisy, image=ctx["control"], prompt_embeds=ctx["prompt"],
            pooled_prompt_embeds=state["pos_pool"], negative_prompt_embeds=state["neg"],
            negative_pooled_prompt_embeds=state["neg_pool"], num_inference_steps=steps,
            generator=gen, controlnet_conditioning_scale=SCALE).images[0]
    sync(torch)
    peak = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
    return image, ms(time.perf_counter() - t0), round(float(peak), 3)


def add_call_hooks(state):
    counts = {"unet": 0, "controlnet": 0}
    handles = []
    for name, module in (("unet", state["pipeline"].unet), ("controlnet", state["pipeline"].controlnet)):
        handles.append(module.register_forward_hook(lambda _m, _a, _o, n=name: counts.__setitem__(n, counts[n] + 1)))
    return counts, handles


def expand_ctx(state, ctx, noisy_batch):
    n = noisy_batch.shape[0]
    return {
        "noisy": noisy_batch,
        "control": ctx["control"].expand(n, -1, -1, -1),
        "prompt": ctx["prompt"].expand(n, -1, -1),
        "pos_pool": state["pos_pool"].expand(n, -1),
        "neg": state["neg"].expand(n, -1, -1),
        "neg_pool": state["neg_pool"].expand(n, -1),
    }


def call_batch(state, ctx, noisy_batch, seeds, steps: int):
    torch = state["torch"]
    ec = expand_ctx(state, ctx, noisy_batch)
    gens = [torch.Generator(device="cuda").manual_seed(int(s)) for s in seeds]
    counts, handles = add_call_hooks(state)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sync(torch)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            images = state["pipeline"](
                latents=ec["noisy"], image=ec["control"], prompt_embeds=ec["prompt"],
                pooled_prompt_embeds=ec["pos_pool"], negative_prompt_embeds=ec["neg"],
                negative_pooled_prompt_embeds=ec["neg_pool"], num_inference_steps=steps,
                generator=gens, controlnet_conditioning_scale=SCALE).images
        sync(torch)
        elapsed = ms(time.perf_counter() - start)
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
        return images, elapsed, round(float(peak), 3), counts, None
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        info = {"error": str(exc), "allocated_mb": float(torch.cuda.memory_allocated() / 1024**2),
                "reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
                "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2)}
        return None, None, info, counts, "oom"
    finally:
        for h in handles:
            h.remove()


def face_metrics(state, ctx, arr: np.ndarray) -> dict[str, object]:
    try:
        face = state["ns"]["analyze_faces"](state["app"], arr)[0]
    except Exception as exc:
        return {"face_detected": False, "identity_similarity_mean": None, "identity_similarity_min": None,
                "identity_error": str(exc)}
    emb = face["embedding"].astype(np.float32)
    emb /= max(np.linalg.norm(emb), 1e-12)
    scores = [float(np.dot(emb, ref)) for ref in ctx["ref_embeddings"]]
    return {"face_detected": True, "identity_similarity_mean": float(np.mean(scores)),
            "identity_similarity_min": float(np.min(scores)), "identity_similarity_to_refs": scores}


def make_sweep_sheet(ctx, paths: dict[int, Path], out: Path) -> None:
    imgs = [Image.open(ctx["group"] / "lq/LQ.png").convert("RGB"), Image.open(ctx["refs"][0]).convert("RGB")]
    labels = ["LQ", "Reference"]
    for s in STEPS:
        imgs.append(Image.open(paths[s]).convert("RGB"))
        labels.append(str(s))
    thumb_w, thumb_h = 256, 256
    sheet = Image.new("RGB", (thumb_w * len(imgs), thumb_h + 32), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(imgs, labels)):
        sheet.paste(im.resize((thumb_w, thumb_h)), (i * thumb_w, 32))
        draw.text((i * thumb_w + 8, 8), label, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def make_batch_sheet(paths: list[Path], seeds: list[int], out: Path) -> None:
    thumb_w, thumb_h = 256, 256
    sheet = Image.new("RGB", (thumb_w * 5, thumb_h * 2 + 32), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, seed) in enumerate(zip(paths, seeds)):
        im = Image.open(p).convert("RGB").resize((thumb_w, thumb_h))
        x, y = (i % 5) * thumb_w, 32 + (i // 5) * thumb_h
        sheet.paste(im, (x, y))
        draw.text((x + 8, 8 + (i // 5) * thumb_h), f"seed={seed}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def run_sweep(base, state, rows):
    results = []
    for idx, row in enumerate(rows):
        ident = f"{row['dataset']}__{row['identity']}"
        log(f"step sweep {idx+1}/{len(rows)}: {ident}")
        ctx = build_identity_context(base, state, row)
        noisy, prep_ms, control = prepare_noisy(state, ctx, SEED)
        ctx["control"] = control
        paths = {}
        for steps in STEPS:
            log(f"  steps={steps}")
            path = OUT / "step_sweep" / ident / f"step_{steps}.png"
            image, infer_ms, peak = call_single(state, ctx, noisy, steps)
            post_ms, arr = postprocess(state, ctx, image, path)
            if anomaly(arr):
                raise StopExperiment(f"invalid/black/NaN output at {ident}, steps={steps}")
            if steps == 4 and anomaly(arr):
                raise StopExperiment(f"4-step output invalid at {ident}")
            paths[steps] = path
            fm = face_metrics(state, ctx, arr)
            results.append({
                "identity": ident, "lq_path": str(ctx["group"] / "lq/LQ.png"),
                "reference_path": str(ctx["refs"][0]), "steps": steps, "seed": SEED,
                "inference_time_ms": infer_ms, "preprocessing_time_ms": prep_ms,
                "identity_feature_extraction_time_ms": ctx["identity_ms"],
                "postprocessing_time_ms": post_ms,
                "total_time_ms": round(prep_ms + ctx["identity_ms"] + infer_ms + post_ms, 3),
                "peak_vram_mb": peak, "output_path": str(path),
                **fm,
            })
        sheet = OUT / "step_sweep" / ident / "contact_sheet_LQ_Reference_1_2_4_6_8_10_50.png"
        make_sweep_sheet(ctx, paths, sheet)
    fields = list(results[0]) if results else []
    with (EXP / "step_sweep_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)
    return results


def run_batch(state, row):
    ident = f"{row['dataset']}__{row['identity']}"
    seeds = list(range(42, 52))
    ctx = build_identity_context(None, state, row)
    noisy_list, prep_total = [], 0.0
    control = None
    for seed in seeds:
        noisy, p, control = prepare_noisy(state, ctx, seed)
        noisy_list.append(noisy)
        prep_total += p
    ctx["control"] = control
    noisy_batch = state["torch"].cat(noisy_list, dim=0)
    prep_total = round(prep_total, 3)
    batch_dir = OUT / "batch_probe" / ident
    batch_dir.mkdir(parents=True, exist_ok=True)
    log("batch probe: attempting true batch=10")
    start_wall = time.perf_counter()
    images, infer_ms, peak, calls, oom = call_batch(state, ctx, noisy_batch, seeds, 4)
    mode = "true_batch_10"
    oom_info = None
    if oom:
        oom_info = peak
        log("batch=10 OOM; falling back to two microbatches of 5")
        state["torch"].cuda.empty_cache()
        images = []
        total_infer = 0.0
        peak_max = 0.0
        calls = {"unet": 0, "controlnet": 0}
        for lo in (0, 5):
            chunk, elapsed, p, c, err = call_batch(state, ctx, noisy_batch[lo:lo+5], seeds[lo:lo+5], 4)
            if err:
                raise StopExperiment(f"microbatch=5 also failed: {p}")
            images.extend(chunk)
            total_infer += elapsed
            peak_max = max(peak_max, p)
            calls["unet"] += c["unet"]; calls["controlnet"] += c["controlnet"]
        infer_ms, peak = total_infer, peak_max
        mode = "microbatch_5_plus_5"
    batch_wall_start = start_wall
    batch_paths, batch_arrays = [], []
    batch_post_ms = 0.0
    for seed, image in zip(seeds, images):
        p = batch_dir / "batch" / f"seed_{seed}.png"
        post, arr = postprocess(state, ctx, image, p)
        batch_post_ms += post
        batch_paths.append(p); batch_arrays.append(arr)
    batch_total_ms = ms(time.perf_counter() - batch_wall_start)
    batch_checks = [face_metrics(state, ctx, arr) for arr in batch_arrays]
    for seed, arr, fm in zip(seeds, batch_arrays, batch_checks):
        if anomaly(arr) or not fm.get("face_detected"):
            raise StopExperiment(f"batch output invalid or face missing at seed={seed}")

    log("batch probe: serial 10 x 4-step generation")
    serial_start = time.perf_counter()
    serial_paths, serial_arrays = [], []
    serial_infer_ms, serial_post_ms, serial_peak = 0.0, 0.0, 0.0
    serial_calls = {"unet": 0, "controlnet": 0}
    for seed, noisy in zip(seeds, noisy_list):
        local_ctx = dict(ctx)
        image, im, pk = call_single_seed(state, local_ctx, noisy, 4, seed)
        serial_infer_ms += im; serial_peak = max(serial_peak, pk)
        serial_calls["unet"] += 1; serial_calls["controlnet"] += 1
        p = batch_dir / "serial" / f"seed_{seed}.png"
        post, arr = postprocess(state, local_ctx, image, p)
        serial_post_ms += post
        serial_paths.append(p); serial_arrays.append(arr)
        if anomaly(arr):
            raise StopExperiment(f"serial 4-step output invalid at seed={seed}")
    serial_total_ms = ms(time.perf_counter() - serial_start)
    serial_checks = [face_metrics(state, ctx, arr) for arr in serial_arrays]

    pairwise = []
    for seed, ba, sa in zip(seeds, batch_arrays, serial_arrays):
        pairwise.append({"seed": seed, "batch_path": str(batch_paths[seeds.index(seed)]),
                         "serial_path": str(serial_paths[seeds.index(seed)]),
                         "mean_abs_diff": float(np.mean(np.abs(ba.astype(np.float32) - sa.astype(np.float32)))),
                         "max_abs_diff": int(np.max(np.abs(ba.astype(np.int16) - sa.astype(np.int16)))),
                         "exact_pixels": bool(np.array_equal(ba, sa)),
                         "batch_face_detected": batch_checks[seeds.index(seed)].get("face_detected"),
                         "serial_face_detected": serial_checks[seeds.index(seed)].get("face_detected")})
    hashes = [sha256(p) for p in batch_paths]
    duplicate_pairs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i] == hashes[j]:
                duplicate_pairs.append([seeds[i], seeds[j]])
    sheet = batch_dir / "batch_contact_sheet_seed_42_51.png"
    make_batch_sheet(batch_paths, seeds, sheet)
    summary = {"identity": ident, "steps": 4, "seeds": seeds, "mode": mode,
               "true_batch_size": 10 if mode == "true_batch_10" else 5,
               "batch_wall_time_ms": batch_total_ms, "batch_inference_time_ms": infer_ms,
               "batch_postprocessing_time_ms": round(batch_post_ms, 3),
               "batch_preprocessing_time_ms": prep_total, "batch_peak_vram_mb": peak,
               "batch_unet_calls": calls["unet"], "batch_controlnet_calls": calls["controlnet"],
               "batch_expected_denoising_rounds": 4,
               "serial_wall_time_ms": serial_total_ms, "serial_inference_time_ms": round(serial_infer_ms, 3),
               "serial_postprocessing_time_ms": round(serial_post_ms, 3), "serial_peak_vram_mb": serial_peak,
               "serial_unet_calls": serial_calls["unet"], "serial_controlnet_calls": serial_calls["controlnet"],
               "wall_clock_saving_ms": round(serial_total_ms - batch_total_ms, 3),
               "wall_clock_saving_ratio": float((serial_total_ms - batch_total_ms) / serial_total_ms),
               "oom": oom_info, "duplicate_pairs": duplicate_pairs, "pairwise_batch_vs_serial": pairwise,
               "outputs": [str(p) for p in batch_paths], "serial_outputs": [str(p) for p in serial_paths],
               "contact_sheet": str(sheet)}
    write_json(EXP / "batch_probe_summary.json", summary)
    rows = []
    for seed, p in zip(seeds, batch_paths):
        rows.append({"identity": ident, "seed": seed, "steps": 4, "mode": mode,
                     "output_path": str(p), "batch_wall_time_ms": batch_total_ms,
                     "batch_peak_vram_mb": peak, "unet_calls": calls["unet"],
                     "controlnet_calls": calls["controlnet"], "serial_path": str(serial_paths[seed - 42]),
                     "mean_abs_diff_vs_serial": pairwise[seed - 42]["mean_abs_diff"],
                     "exact_pixels_vs_serial": pairwise[seed - 42]["exact_pixels"]})
    with (EXP / "batch_probe_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return summary


def call_single_seed(state, ctx, noisy, steps: int, seed: int):
    torch = state["torch"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sync(torch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        image = state["pipeline"](
            latents=noisy, image=ctx["control"], prompt_embeds=ctx["prompt"],
            pooled_prompt_embeds=state["pos_pool"], negative_prompt_embeds=state["neg"],
            negative_pooled_prompt_embeds=state["neg_pool"], num_inference_steps=steps,
            generator=gen, controlnet_conditioning_scale=SCALE).images[0]
    sync(torch)
    peak = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
    return image, ms(time.perf_counter() - start), round(float(peak), 3)


def main() -> int:
    started = time.perf_counter()
    try:
        rows = read_selected_rows()
        selection = [{"dataset": r["dataset"], "identity": r["identity"], "split": r["split"],
                      "source_set": r.get("source_set"), "group_index": r.get("group_index"),
                      "inputs": input_records(r)} for r in rows]
        write_json(EXP / "selection_manifest.json", {"selection_basis": "frozen split train/development only; complete 512x512 RGB LQ + 3 refs; source and historical-output coverage", "rows": selection, "frozen_split": str(FROZEN), "final20_manifest_checked": str(FINAL20)})
        write_json(EXP / "config_snapshot.json", {"steps": STEPS, "seed": SEED, "batch_seeds": list(range(42, 52)), "prompt": PROMPT, "negative_prompt": NEG_PROMPT, "controlnet_conditioning_scale": SCALE, "model": str(PROJECT / MODEL), "controlnet": str(PROJECT / CONTROLNET), "mix": str(PROJECT / MIX), "photomaker": str(PHOTO), "source_infer": str(SOURCE_INFER), "reused_runner": str(RUNNER_DIR / "run_teacher_outputs.py"), "no_mria": True, "no_idr": True, "color_correction": True, "fp16": True, "sequential_cpu_offload": True, "attention_slicing": "max", "vae_slicing": True, "vae_tiling": True, "insightface_provider": "CPUExecutionProvider"})
        log("starting isolated FaceMe probe")
        base, state = load_runtime()
        torch = state["torch"]
        log(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
        # Explicit warmup, excluded from formal CSV.
        warm_ctx = build_identity_context(base, state, rows[0])
        warm_noisy, _, warm_control = prepare_noisy(state, warm_ctx, SEED)
        warm_ctx["control"] = warm_control
        _, warm_ms, warm_peak = call_single(state, warm_ctx, warm_noisy, 1)
        write_json(EXP / "warmup.json", {"steps": 1, "seed": SEED, "time_ms": warm_ms, "peak_vram_mb": warm_peak, "excluded_from_formal_results": True})
        log(f"warmup complete: {warm_ms} ms, peak={warm_peak} MB")
        sweep = run_sweep(base, state, rows)
        summary = run_batch(state, rows[0])
        write_json(EXP / "run_summary.json", {"status": "completed", "elapsed_ms": ms(time.perf_counter() - started), "sweep_rows": len(sweep), "batch_mode": summary["mode"], "batch_summary": str(EXP / "batch_probe_summary.json")})
        log("probe completed")
        return 0
    except StopExperiment as exc:
        log("STOP CONDITION: " + str(exc))
        write_json(EXP / "run_summary.json", {"status": "stopped_by_stop_condition", "reason": str(exc), "elapsed_ms": ms(time.perf_counter() - started)})
        return 2
    except Exception as exc:
        log("FAILED: " + repr(exc))
        traceback.print_exc()
        write_json(EXP / "run_summary.json", {"status": "failed", "error": repr(exc), "elapsed_ms": ms(time.perf_counter() - started)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
