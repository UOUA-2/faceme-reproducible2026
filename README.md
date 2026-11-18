# FaceMe / ICVIP reproducible source

This repository contains the FaceMe/ICVIP source code, runtime configuration, dependency files, and reproducibility manifests. It intentionally does not contain local Python environments, download caches, generated results, private experiment records, or large model files.

## Model and weight inventory

| Component | Required path | Official download? | Action after cloning |
|---|---|---|---|
| RealVisXL V3.0 | `models/SG161222--RealVisXL_V3.0/` | Yes: [SG161222/RealVisXL_V3.0](https://huggingface.co/SG161222/RealVisXL_V3.0) | Download the model with Hugging Face, or place the local snapshot at this path. |
| PhotoMaker v1 | `models/TencentARC--PhotoMaker/photomaker-v1.bin` | Yes: [TencentARC/PhotoMaker](https://huggingface.co/TencentARC/PhotoMaker) | Download `photomaker-v1.bin` from the official model repository. |
| InsightFace antelopev2 | `models/antelopev2/` | Yes: [official InsightFace model zoo](https://github.com/deepinsight/insightface/releases) | Download the official `antelopev2` pack, extract it here, and follow its license. |
| FaceMe ControlNet | `weights/controlnet/` | No verified public replacement found | Preserve this directory in a separate backup for exact FaceMe reproduction. |
| FaceMe mix / adapter | `weights/mix/mix.safetensors` | No verified public replacement found | Preserve this file in a separate backup for exact FaceMe reproduction. |

The first three components are external official dependencies and are not duplicated here. The ControlNet and `mix.safetensors` entries are project-specific artifacts: the source code refers to them by local path, but this repository does not establish an official public download location for them. Do not delete those two items until a separate backup has been verified.

## Reproduction notes

1. Clone this repository and create a clean Python environment using `requirements_compatible.txt` as the starting point.
2. Download the three official dependencies above into the required paths.
3. Restore the project-specific `weights/controlnet/` and `weights/mix/mix.safetensors` from a separate backup.
4. Place input data under the paths expected by the selected experiment and run the corresponding script under `experiments/`.

`requirements.txt` records a later local environment snapshot. `requirements_compatible.txt` is the conservative installation proposal for the FaceMe runtime; verify CUDA and provider imports after installation.

## What is intentionally excluded

Model caches, pip caches, Conda/venv environments, generated images, logs, reports, temporary archives, and large datasets are excluded. `reproducibility/MODEL_MANIFEST.tsv` and `reproducibility/DATA_MANIFEST.tsv` record the excluded local inventory so it can be audited or re-obtained later.
