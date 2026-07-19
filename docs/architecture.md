# ReliPose-HOI Compact Architecture

This repository is intentionally a compact research implementation. It
keeps the paper behavior and removes framework-style machinery that is
not needed for the first real Colab training run.

## Scientific Path

```text
image
-> one shared FPN backbone
-> detector branch using shared FPN features
-> Human/Object/Union RoI extraction
-> internal sparse pose tokenizer
-> pose-quality-guided sparse refinement
-> compact anatomy and human-level joint representations
-> object-independent Joint Reliability r_hi
-> human-object pair feature q_ho
-> pair-dependent Interaction Relevance a_hoi
-> g_hoi = r_hi * a_hoi
-> gated joint attention
-> adaptive residual pose-visual fusion
-> 117 verb logits
-> deterministic object-conditioned 600-HOI scores
```

`rho_hi` is the coarse pose quality used by sparse refinement. It is not
Joint Reliability. `r_hi` is computed once per human and joint and has no
object or pair inputs. `a_hoi` is computed after pair expansion. The gate
`g_hoi` is the exact elementwise product `r_hi * a_hoi`.

## Data

COCO Keypoints provides person boxes and keypoints for internal pose
pretraining. HICO-DET provides images, human boxes, object boxes, object
labels, positive pair indices and multi-hot verb/HOI targets. HICO
targets contain no external pose predictions, keypoint cache, Joint
Reliability, Interaction Relevance or gates.

## Training

The supported scripts cover:

- `pose`: COCO pose pretraining using GT person boxes.
- `hoi_oracle`: HICO development using Oracle/GT boxes. This is
  diagnostic and not standard detected-box evaluation.
- `hoi_detected`: inference/training path using the shared-FPN detector.
- `robust`: clean/corrupted training with object-independent pose
  corruption and synthetic Reliability supervision.
- `finetune`: same compact machinery with lower learning rates.

The optimizer is AdamW. The scheduler is cosine decay with optional
warmup. Checkpoints store model, optimizer, scheduler, epoch, global
step, configuration and random states.

## Postponed

Final official HICO mAP, visualization dashboards, ablation frameworks,
DDP support, TensorBoard/cloud logging and MATLAB export are deliberately
postponed. They can be added later on top of the compact prediction
outputs without changing the architecture.

