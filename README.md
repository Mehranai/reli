# ReliPose-HOI

Compact PyTorch research implementation of ReliPose-HOI: integrated
sparse pose reasoning with decoupled object-independent Joint
Reliability and pair-dependent Interaction Relevance for robust
human-object interaction detection.

The code is designed for paper reproduction and Colab training, not as a
general ML framework.

## What Is Preserved

- one shared visual FPN backbone;
- Human/Object/Union RoI extraction;
- internal sparse pose estimation from Human RoI features;
- pose-quality-guided sparse refinement with `rho_hi`;
- compact anatomy and object-independent Joint Reliability `r_hi`;
- pair feature `q_ho`;
- pair-dependent Interaction Relevance `a_hoi`;
- exact `g_hoi = r_hi * a_hoi`;
- gated attention and adaptive residual pose-visual fusion;
- 117 learned multi-label verb logits;
- deterministic object-conditioned projection to 600 HICO scores;
- COCO pose pretraining, HICO Oracle-box development, HICO detected-box
  workflow, robust corruption training and checkpoint resume.

There is no ViPLO dependency, no external pose estimator, no precomputed
HICO pose input and no learned 600-class classifier.

## Install

```bash
pip install -e .
```

For Colab training, install PyTorch and torchvision matching the runtime.
This compact code does not download datasets or pretrained weights.

## Data

Set local paths in one of:

```text
configs/pose.yaml
configs/hoi_oracle.yaml
configs/hoi_detected.yaml
configs/robust.yaml
```

COCO Keypoints is used only for pose pretraining. HICO targets contain
boxes, labels and HOI annotations only; pose is always internal.

## Commands

Smoke test:

```bash
python scripts/smoke_test.py --config configs/smoke.yaml
```

Training:

```bash
python scripts/train.py --config configs/pose.yaml --stage pose
python scripts/train.py --config configs/hoi_oracle.yaml --stage hoi_oracle
python scripts/train.py --config configs/hoi_detected.yaml --stage hoi_detected
python scripts/train.py --config configs/robust.yaml --stage robust
```

Oracle-box runs are diagnostic/non-standard. Detected-box experiments use
the detector branch that consumes the already-computed shared FPN maps.

## Architecture

See `docs/architecture.md`. The original research specification remains
in `docs/research_spec.md`.

