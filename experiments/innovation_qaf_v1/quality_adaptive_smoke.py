from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2


PROJECT = Path('/mnt/f/ICVIP/DaChuang')
CONFIG_PATH = PROJECT / 'experiments/innovation_qaf_v1/qaf_config.json'
OUTPUT_ROOT = PROJECT / 'outputs/innovation_smoke_v1'
RUNTIME_HELPER = PROJECT / 'experiments/innovation_qaf_v1/runtime_baseline_offload.py'
SOURCE_INFER = PROJECT / 'infer.py'
SOURCE_DEMO = PROJECT / 'demo.py'
BASELINE_DIR = PROJECT / 'sample_output_trial_resized_cc/LFW-Test-Ref__Lloyd_Mudiwa_0001'
CALL_RECORDS: list[dict] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def image_info(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot decode image: {path}')
    return {
        'path': str(path),
        'sha256': sha256(path),
        'bytes': path.stat().st_size,
        'width': int(image.shape[1]),
        'height': int(image.shape[0]),
    }


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def interface_audit() -> dict:
    infer = SOURCE_INFER.read_text(encoding='utf-8')
    demo = SOURCE_DEMO.read_text(encoding='utf-8')
    checks = {
        'infer_input_dir_is_lq': 'args.input_dir' in infer,
        'infer_ref_dir_is_identity_input': 'args.ref_dir' in infer,
        'infer_fixed_steps_50': 'num_inference_steps=50' in infer,
        'infer_has_no_strength_cli': 'controlnet_conditioning_scale' not in infer,
        'demo_strength_parameter_exists': 'strength: float' in demo,
        'demo_strength_maps_to_controlnet': 'controlnet_conditioning_scale=float(strength)' in demo,
        'demo_cfg_maps_to_guidance': 'guidance_scale=float(cfg_scale)' in demo,
        'demo_ref_embeddings_use_mix': 'embs = mix(clip_emb=ref_clip_embs' in demo,
    }
    if not all(checks.values()):
        raise RuntimeError(f'Interface audit failed: {checks}')
    return checks


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(config: dict) -> dict:
    import torch
    import diffusers

    os.chdir(PROJECT)
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))

    checks = interface_audit()
    lq = Path(config['input'])
    refs = [Path(item) for item in config['refs']]
    baseline_output = Path(config['baseline_output'])
    baseline_manifest = json.loads((BASELINE_DIR / 'TRIAL_MANIFEST.json').read_text(encoding='utf-8'))
    baseline_input = Path(baseline_manifest['resized_lq'])
    if sha256(lq) != sha256(baseline_input):
        raise RuntimeError('Innovation input hash does not match the baseline input hash.')

    gray = cv2.imread(str(lq), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f'Cannot read LQ input: {lq}')
    rule = config['rule']
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = float(min(laplacian_variance / float(rule['sharpness_normalizer']), 1.0))
    degradation = float(1.0 - sharpness)
    strength = float(min(max(
        float(rule['base_controlnet_strength']) + float(rule['adaptation_alpha']) * degradation,
        float(rule['min_controlnet_strength']),
    ), float(rule['max_controlnet_strength'])))

    helper = load_module(RUNTIME_HELPER, 'qaf_runtime_baseline_offload')
    helper.OUTPUT_ROOT = OUTPUT_ROOT
    helper.COLOR_CORRECTION = True

    original_call = diffusers.StableDiffusionXLControlNetPipeline.__call__

    def quality_adaptive_call(self, *args, **kwargs):
        kwargs.setdefault('controlnet_conditioning_scale', strength)
        CALL_RECORDS.append({
            'controlnet_conditioning_scale': float(kwargs['controlnet_conditioning_scale']),
            'guidance_scale_present': 'guidance_scale' in kwargs,
            'num_inference_steps': kwargs.get('num_inference_steps'),
        })
        return original_call(self, *args, **kwargs)

    diffusers.StableDiffusionXLControlNetPipeline.__call__ = quality_adaptive_call
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    group = {
        'dataset': 'LFW-Test-Ref',
        'identity': 'Lloyd_Mudiwa_0001',
        'lq': str(lq),
        'refs': [str(item) for item in refs],
    }
    start = time.monotonic()
    try:
        result = helper.run_offloaded_infer(group, lq.parent, refs[0].parent)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        diffusers.StableDiffusionXLControlNetPipeline.__call__ = original_call
    elapsed = time.monotonic() - start
    output = OUTPUT_ROOT / lq.name
    result.update({
        'elapsed_seconds': elapsed,
        'qaf_records': CALL_RECORDS,
        'input': image_info(lq),
        'refs': [image_info(item) for item in refs],
        'baseline_input': image_info(baseline_input),
        'baseline_output': image_info(baseline_output),
        'same_input_as_baseline': sha256(lq) == sha256(baseline_input),
        'innovation_output': image_info(output),
        'interface_audit': checks,
        'adaptive_rule': {
            'laplacian_variance': laplacian_variance,
            'sharpness_score': sharpness,
            'degradation_score': degradation,
            'final_controlnet_conditioning_scale': strength,
            'formula': rule['formula'],
            'bounded_range': [rule['min_controlnet_strength'], rule['max_controlnet_strength']],
        },
        'source_hashes': {
            'infer.py': sha256(SOURCE_INFER),
            'demo.py': sha256(SOURCE_DEMO),
        },
        'runtime': {
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'torch_cuda_available': bool(torch.cuda.is_available()),
            'seed': 42,
            'steps': 50,
            'unet_variant': 'fp16',
            'sequential_cpu_offload': True,
            'attention_slicing': 'max',
            'vae_slicing': True,
            'vae_tiling': True,
            'insightface_providers': ['CPUExecutionProvider'],
            'color_correction': True,
            'torch_peak_memory_allocated_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            'torch_peak_memory_reserved_bytes': int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        },
        'ref_fusion_note': 'REF uses the unchanged baseline ref_id_embs/ref_clip_embs -> Mix path; no unsupported REF scale was added.',
        'validation_note': 'One smoke run only; this is not evidence that the rule is effective.',
    })
    return result


def run() -> int:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError(f'Refusing to overwrite existing output: {OUTPUT_ROOT}')
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    lq = Path(config['input'])
    refs = [Path(item) for item in config['refs']]
    for path in [lq, *refs, Path(config['baseline_output']), RUNTIME_HELPER]:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / 'qaf_config.json')
    report_path = OUTPUT_ROOT / 'QAF_RUN_REPORT.json'
    log_path = OUTPUT_ROOT / 'run.log'
    report = {
        'experiment': 'quality_adaptive_fusion_v1',
        'status': 'single_group_smoke_only_not_validated',
        'config_path': str(CONFIG_PATH),
        'output_dir': str(OUTPUT_ROOT),
        'no_training': True,
        'no_batch_run': True,
    }
    with log_path.open('w', encoding='utf-8') as log_handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_handle)), contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):
            try:
                report.update(execute(config))
            except Exception as exc:
                report.update({'success': False, 'error_type': type(exc).__name__, 'error': str(exc)})
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                raise
            else:
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                shutil.copy2(Path(__file__), OUTPUT_ROOT / 'quality_adaptive_smoke.py')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get('success') else 1


if __name__ == '__main__':
    raise SystemExit(run())
