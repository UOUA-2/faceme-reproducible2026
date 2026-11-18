from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2


PROJECT = Path('/mnt/f/ICVIP/DaChuang')
SAMPLE_ROOT = PROJECT / 'sample_input'
OUTPUT_ROOT = PROJECT / 'sample_output_inferred'
COLOR_CORRECTION = False
SOURCE_INFER = PROJECT / 'infer.py'
MAPPING_PATH = SAMPLE_ROOT / 'inferred_mapping.json'

GROUPS = [
    ('LFW-Test-Ref', 'Madeleine_Webber_0001'),
    ('LFW-Test-Ref', 'Lloyd_Mudiwa_0001'),
    ('Wider-Test-Ref', '0481'),
    ('Wider-Test-Ref', '0740'),
    ('WebPhoto-Test-Ref', '00011_00'),
    ('WebPhoto-Test-Ref', '00123_00'),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def score(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f'Cannot decode {path}')
    variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return {
        'path': str(path),
        'relative_to_project': str(path.relative_to(PROJECT)),
        'filename': path.name,
        'width': int(image.shape[1]),
        'height': int(image.shape[0]),
        'bytes': path.stat().st_size,
        'sha256': sha256(path),
        'laplacian_variance': variance,
    }


def build_mapping() -> tuple[dict, dict]:
    groups = []
    first_lfw = None
    for dataset, identity in GROUPS:
        directory = SAMPLE_ROOT / dataset / identity
        items = [score(directory / f'{index}.png') for index in range(4)]
        lowest = min(items, key=lambda item: item['laplacian_variance'])
        for item in items:
            item['inferred_role'] = 'LQ' if item is lowest else 'REF'
            item['official_role_confirmed'] = False
            item['inference_role_basis'] = 'lowest Laplacian variance => inferred LQ; all other images => inferred REF'
        group = {
            'dataset': dataset,
            'identity': identity,
            'identity_directory': str(directory),
            'role_mapping_status': 'INFERRED_NOT_OFFICIAL',
            'lq': lowest['path'],
            'refs': [item['path'] for item in items if item is not lowest],
            'files': items,
            'official_mapping_confirmed': False,
            'supports_multiple_refs_in_infer_py': True,
        }
        groups.append(group)
        if first_lfw is None and dataset == 'LFW-Test-Ref':
            first_lfw = group

    mapping = {
        'purpose': 'inferred sample-role mapping for one controlled test; not official annotation',
        'official_role_confirmed': False,
        'inference_method': 'minimum Laplacian variance per identity directory is provisionally LQ; remaining three are provisionally REF',
        'source_directory': str(SAMPLE_ROOT),
        'groups': groups,
        'selected_test': {
            'dataset': first_lfw['dataset'],
            'identity': first_lfw['identity'],
            'lq': first_lfw['lq'],
            'refs': first_lfw['refs'],
            'reason': 'first LFW identity in the existing priority order',
        },
        'infer_py_interface_check': {
            'supports_one_lq_multiple_refs': True,
            'evidence': 'infer.py lists every file in ref_dir, computes one embedding per file, then concatenates ref_id_embs and ref_clip_embs before Mix.',
            'single_ref_fallback_used': False,
        },
    }
    return mapping, first_lfw


def stage_test_inputs(group: dict) -> tuple[Path, Path]:
    stage = SAMPLE_ROOT / 'inferred_run' / group['dataset'] / group['identity']
    lq_dir = stage / 'lq'
    ref_dir = stage / 'ref'
    lq_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    source_lq = Path(group['lq'])
    destination_lq = lq_dir / source_lq.name
    destination_lq.write_bytes(source_lq.read_bytes())
    for source in group['refs']:
        source_path = Path(source)
        (ref_dir / source_path.name).write_bytes(source_path.read_bytes())
    return lq_dir, ref_dir


def run_offloaded_infer(group: dict, lq_dir: Path, ref_dir: Path) -> dict:
    import torch
    import transformers

    original_source = SOURCE_INFER.read_text(encoding='utf-8')
    original_hash = hashlib.sha256(original_source.encode('utf-8')).hexdigest()
    direct_moves = """    vae.to(device, dtype=torch.float16)
    unet.to(device, dtype=weight_dtype)
    mix.to(device, dtype=weight_dtype)
    id_encoder_clip.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)
"""
    offload_moves = """    # Temporary runtime wrapper: pipeline modules use sequential CPU offload.
    mix.to(device, dtype=weight_dtype)
    id_encoder_clip.to(device, dtype=weight_dtype)
"""
    if direct_moves not in original_source:
        raise RuntimeError('Expected original model move block was not found; refusing to run.')
    source = original_source.replace(direct_moves, offload_moves, 1)
    pipeline_move = '    pipeline = pipeline.to(device=device)\n'
    offload_setup = """    # Temporary low-memory runtime settings; original infer.py remains unchanged.
    pipeline.enable_sequential_cpu_offload()
    pipeline.enable_attention_slicing('max')
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
"""
    if pipeline_move not in source:
        raise RuntimeError('Expected pipeline move block was not found; refusing to run.')
    source = source.replace(pipeline_move, offload_setup, 1)

    namespace = {'__name__': 'inferred_lfw_runtime', '__file__': str(SOURCE_INFER)}
    exec(compile(source, str(SOURCE_INFER), 'exec'), namespace)

    cache = {}

    def patch_lowmem(cls, force_variant=None):
        original = cls.from_pretrained.__func__

        def loader(c, *args, **kwargs):
            kwargs.setdefault('low_cpu_mem_usage', True)
            kwargs.setdefault('torch_dtype', torch.float16)
            if force_variant is not None:
                kwargs['variant'] = force_variant
            obj = original(c, *args, **kwargs)
            subfolder = kwargs.get('subfolder')
            if subfolder:
                cache[subfolder] = obj
            return obj

        cls.from_pretrained = classmethod(loader)

    patch_lowmem(namespace['OriginalUNet2DConditionModel'], force_variant='fp16')
    patch_lowmem(namespace['AutoencoderKL'])
    patch_lowmem(namespace['ControlNetModel'])
    for name in ('CLIPTextModel', 'CLIPTextModelWithProjection'):
        if hasattr(transformers, name):
            patch_lowmem(getattr(transformers, name))

    pipeline_class = namespace['StableDiffusionXLControlNetPipeline']
    original_pipeline_loader = pipeline_class.from_pretrained.__func__

    def pipeline_loader(c, *args, **kwargs):
        if 'text_encoder' not in kwargs and 'text_encoder' in cache:
            kwargs['text_encoder'] = cache['text_encoder']
        if 'text_encoder_2' not in kwargs and 'text_encoder_2' in cache:
            kwargs['text_encoder_2'] = cache['text_encoder_2']
        return original_pipeline_loader(c, *args, **kwargs)

    pipeline_class.from_pretrained = classmethod(pipeline_loader)

    original_face_analysis = namespace['FaceAnalysis2']

    class CPUFaceAnalysis2(original_face_analysis):
        def __init__(self, *args, **kwargs):
            kwargs['providers'] = ['CPUExecutionProvider']
            super().__init__(*args, **kwargs)

    namespace['FaceAnalysis2'] = CPUFaceAnalysis2
    local_photomaker = PROJECT / 'models/TencentARC--PhotoMaker/photomaker-v1.bin'
    namespace['hf_hub_download'] = lambda *args, **kwargs: str(local_photomaker)

    args = namespace['argparse'].Namespace(
        pretrained_model_name_or_path='models/SG161222--RealVisXL_V3.0',
        controlnet_model_name_or_path='weights/controlnet',
        mix_path='weights/mix',
        seed=42,
        input_dir=str(lq_dir),
        result_dir=str(OUTPUT_ROOT),
        ref_dir=str(ref_dir),
        pos_prompt='A photo of face.',
        neg_prompt='A photo of face.',
        key_word='face',
        color_correction=COLOR_CORRECTION,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    namespace['main'](args)
    elapsed = time.monotonic() - start
    output_files = sorted(str(path) for path in OUTPUT_ROOT.iterdir() if path.is_file())
    return {
        'success': bool(output_files),
        'elapsed_seconds': elapsed,
        'output_files': output_files,
        'runtime': {
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'torch_cuda_available': bool(torch.cuda.is_available()),
            'seed': 42,
            'unet_variant': 'fp16',
            'sequential_cpu_offload': True,
            'attention_slicing': 'max',
            'vae_slicing': True,
            'vae_tiling': True,
            'insightface_providers': ['CPUExecutionProvider'],
            'original_infer_sha256_before_run': original_hash,
        },
    }


def main() -> None:
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    mapping, first_lfw = build_mapping()
    MAPPING_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    lq_dir, ref_dir = stage_test_inputs(first_lfw)
    run_report = {
        'mapping': str(MAPPING_PATH),
        'selected_group': {'dataset': first_lfw['dataset'], 'identity': first_lfw['identity']},
        'staged_lq_dir': str(lq_dir),
        'staged_ref_dir': str(ref_dir),
        'output_dir': str(OUTPUT_ROOT),
        'official_role_confirmed': False,
    }
    try:
        run_report.update(run_offloaded_infer(first_lfw, lq_dir, ref_dir))
    except Exception as exc:
        run_report.update({'success': False, 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    (OUTPUT_ROOT / 'INFER_RUN_REPORT.json').write_text(json.dumps(run_report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(run_report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
