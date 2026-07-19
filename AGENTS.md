# ReliPose-HOI Development Instructions

## Project Goal

Build a complete research-grade PyTorch implementation of:

ReliPose-HOI: Integrated Sparse Pose Reasoning with Decoupled Joint
Reliability and Interaction Relevance for Robust Human-Object
Interaction Detection.

This is a new standalone project. Do not reuse ViPLO, PViC, QPIC, UPT,
or another HOI repository as the implementation base.

Existing papers and repositories may be consulted only for conceptual
comparison. Their source code must not be copied into this project.

## Sources of Truth

Use the following priority order:

1. `docs/research_spec.md`
2. This `AGENTS.md`
3. Tensor and API contracts created under `docs/`
4. `reference/relipose_prototype.ipynb`

The notebook is an exploratory prototype, not production code.
Extract useful ideas and tensor contracts from it, but redesign the
implementation as small, tested Python modules.

If the notebook conflicts with `docs/research_spec.md`, follow
`docs/research_spec.md`.

## Mandatory Architecture

The model must be implemented from scratch as a standalone,
detector-agnostic PyTorch project.

Required pipeline:

1. Input image
2. Shared visual backbone
3. Human and object detection branch
4. Human RoI feature extraction
5. Integrated Sparse Pose Tokenizer
6. Pose-quality-guided sparse refinement
7. Compact anatomical feature extraction
8. Object-independent Joint Reliability
9. Human-object pair encoding
10. Pair-dependent Interaction Relevance
11. Reliability-Relevance gated cross-attention
12. Adaptive residual pose-visual fusion
13. Multi-label HOI prediction

## Internal Pose Module

Pose estimation is an internal component of the architecture.

Do not:

- require MMPose;
- require ViTPose as a separate inference model;
- load precomputed HICO-DET keypoints;
- store pose outputs as required dataset fields;
- run a second independent image backbone for pose;
- use random pose tensors in production code.

The internal pose module must use the shared visual backbone and Human
RoI features.

It must produce:

- joint tokens;
- normalized joint coordinates;
- joint confidence;
- joint uncertainty;
- coarse pose quality.

The default skeleton has 17 COCO joints, but the implementation should
be configurable.

## Scientific Variable Separation

Use the following notation and semantics consistently.

### Coarse Pose Quality

`rho_hi`

- computed inside the pose branch;
- used to guide sparse joint refinement;
- not the final Joint Reliability;
- object-independent.

### Joint Reliability

`r_hi`

- computed once per human and joint;
- measures whether the joint representation is trustworthy;
- object-independent;
- must not receive object or human-object-pair information.

Reliability code must never accept:

- object boxes;
- object classes;
- object features;
- pair features;
- interaction labels;
- joint-object geometry.

### Interaction Relevance

`a_hoi`

- computed for every human-object pair and joint;
- pair-dependent;
- may use object semantics, pair features and joint-object geometry.

### Final Gate

`g_hoi = r_hi * a_hoi`

Reliability and Relevance must remain separately inspectable outputs.

## Refinement

Training should support a differentiable soft refinement gate.

Inference should support hard Top-K refinement of uncertain joints.

Residual refinement heads must use small nonzero initialization.
Do not zero-initialize them in a way that blocks gradients to upstream
refinement layers.

## Datasets

### COCO Keypoints

Used to pretrain the internal pose branch.

### HICO-DET

Used for HOI training and evaluation.

The HICO-DET Dataset class must return images, detection/HOI
annotations and targets. It must not return precomputed pose
predictions.

Dataset paths must come from configuration or command-line arguments.

Unit tests must not download or require either dataset.

## Training Stages

Support these stages:

1. Pose pretraining on COCO Keypoints
2. HOI training with the pose module frozen
3. Controlled joint fine-tuning
4. Full evaluation on HICO-DET

Training configuration must support freezing and unfreezing:

- visual backbone;
- detector;
- pose tokenizer;
- sparse refiner;
- ReliPose reasoning modules;
- HOI head.

## Losses

Support:

- pose coordinate loss;
- pose confidence loss;
- HOI multi-label loss;
- synthetic Joint Reliability loss;
- clean/corrupted prediction consistency loss;
- sparse-refinement regularization.

HOI prediction is multi-label and must use sigmoid-compatible losses.
Do not use a single-class Softmax over verbs.

## Robust Training

Implement training-only pose corruption with configurable operations:

- coordinate jitter;
- joint dropout;
- limb dropout;
- side dropout;
- left-right swap;
- confidence corruption;
- uncertainty corruption;
- bone-length distortion;
- out-of-body displacement;
- joint-token corruption.

Corrupted joints may create synthetic or pseudo Reliability targets.
Uncorrupted pose predictions must not be described as guaranteed
ground-truth Reliability.

## Engineering Requirements

- Python 3.10 or newer.
- PyTorch and torchvision.
- `src/` package layout.
- No notebook-only implementation.
- No global `CFG` singleton inside library modules.
- Structured dataclass or YAML configuration.
- Type hints for public APIs.
- Shape-oriented docstrings.
- Small cohesive modules.
- CPU-compatible unit tests.
- CUDA must be optional.
- No network access during tests.
- No hard-coded local or Colab paths.
- No random placeholder features in production code.
- Support empty-human and empty-pair cases.
- Avoid monolithic files.
- Do not silently catch important errors.
- Validate tensor shapes at public module boundaries.
- Use deterministic seeds in tests.
- Keep optional dataset dependencies outside core package imports.

## Testing Requirements

Every major module must have unit tests for:

- tensor shapes;
- finite outputs;
- empty inputs;
- deterministic evaluation;
- value-range constraints;
- gradient flow;
- invalid-input handling.

The full project must include:

- unit tests;
- a CPU smoke test;
- a tiny end-to-end forward/backward integration test;
- optional real-data integration tests;
- a minimal Colab launcher notebook.

## Colab Role

Google Colab is only the runtime environment for:

- cloning the repository;
- installing it;
- mounting or downloading datasets;
- starting pose pretraining;
- starting HICO-DET training;
- resuming checkpoints;
- evaluation and visualization.

Production classes and training logic must remain in Python modules,
not duplicated in notebook cells.

## Workflow Rules

Before editing:

1. Read this file.
2. Read `docs/research_spec.md`.
3. Inspect the prototype notebook.
4. Inspect the existing repository.
5. State the implementation plan.

During implementation:

- work in small phases;
- run relevant tests after every phase;
- fix failures before continuing;
- update documentation when tensor contracts change;
- do not change the scientific architecture silently.

At the end of every task, report:

- files created;
- files modified;
- commands executed;
- exact test results;
- unresolved issues;
- architecture deviations.

Do not commit changes unless explicitly instructed.
Do not claim a test passed unless it was actually executed.