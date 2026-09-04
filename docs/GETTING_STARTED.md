# Getting started with v3b_v3

[Back to the project overview](../README.md)

Run all commands below from the repository root.

`v3b_v3` is the complete, single-path pipeline used by the accepted three-room
regression.  The public input is a directory of overlapping room photographs;
the final outputs are a Unity-ready asset directory and an importable
`.unitypackage`.

The repository contains the orchestration and reconstruction/material code. It
does **not** redistribute datasets, third-party repositories, or model weights.
Configure those external dependencies before the first run.

The tested path targets Linux with an NVIDIA CUDA GPU. An 11 GiB RTX 2080 Ti
is supported through the automatic DA3 low-memory path; more VRAM is useful for
rooms with many source views. Keep several gigabytes of free disk space per
room: intermediate depth, masks, projected atlases, and audit artifacts are
retained so an interrupted run can be resumed and its decisions inspected.

## What is fixed in v3b_v3

- One pipeline is used for every room. There are no L-room, Structure3D,
  face-name, fixed-corner-count, or fixed-material-count routes.
- Lab clustering only proposes fine spatial regions. It never decides that two
  regions are the same material.
- MatSeg is used only for same/different material identity. Its primary evidence
  is measured in reverse-projected original photographs; a contact-sheet pass
  only resolves the documented two-singleton ambiguity.
- Each material is selected by maximum atlas projection mass, traced back to an
  original image, geometrically rectified, and passed to unchanged CHORD.
- Every 512x512 trace-back input is inferred with the same CHORD configuration
  at its original size. Outputs from different inference sizes are never mixed.
- The accepted v1 layout backend expands each material into a whole territory.
  PBR synthesis preserves the measured texture scale and does not use a fixed
  tile grid or rectangular patch cut-and-paste.
- BaseColor, normal, roughness, and metallic use the same spatial mapping.

## Input contract

Create a dataset directory containing:

```text
my_room/
  input_images/
    0000.jpg
    0001.jpg
    ...
```

Optional files:

- `camera_metadata.json` for known same-center cameras, including panorama-derived synthetic views.
- `existing_da3/results.npz`, `existing_da3/scene.glb`, and
  `existing_da3/camera_poses.json` to reuse a compatible DA3 inference.

A COLMAP model is optional. The default DA3-aligned projection path does not
require one.

The optional known-camera schema is documented in
[camera metadata](CAMERA_METADATA.md).

## Setup

1. Install the external components listed in [THIRD_PARTY.md](../THIRD_PARTY.md).
2. Install the ordinary runtime libraries in the reconstruction environment:

   ```bash
   python -m pip install -r requirements-runtime.txt
   ```

3. Copy `config/v3b.env.example` to `config/v3b.env`.
4. Set the dataset, Python-environment, repository, and checkpoint paths.

Separate Python environments are recommended because DA3, RoomFormer, SAM3,
MatSeg, CHORD, and LaMa have conflicting dependencies.

## Run

```bash
bash run_from_images.sh
```

To use a config outside the repository:

```bash
CONFIG_FILE=/absolute/path/to/room.env bash run_from_images.sh
```

Interrupted jobs can be continued without replacing completed stages:

```bash
RESUME=1 CONFIG_FILE=/absolute/path/to/room.env bash run_from_images.sh
```

The eight stages are `frontend_reconstruction`, `unified_proposals`,
`material_identity_traceback`, `chord_pbr`, `material_layout`, `territory_pbr`,
`unity_project`, and `unitypackage`. `RUN_FROM` and `RUN_UNTIL` may delimit a
diagnostic or resumed run.

## Outputs

Under `outputs/<RUN_NAME>/`:

- `unified_material_frontend/identity/identity_receipt.json`: proof of the
  MatSeg identity-only contract.
- `unified_material_frontend/final_trace/material_level_traceback.json`:
  maximum-weight atlas selection and original-image trace-back provenance.
- `pbr_full_normalized/`: final per-face PBR maps, preview, and scale/lattice
  audits.
- `unity_project/`: OBJ, MTL, BaseColor, normal, roughness, metallic,
  Unity-packed metallic/smoothness, and an Editor importer.
- `v3b_v3.unitypackage`: the importable Unity package.
- `v3b_v3_end_to_end_manifest.json`: hashes linking the input-stage contracts
  to the exported package.

## Unity import

In Unity, choose **Assets > Import Package > Custom Package** and select
`v3b_v3.unitypackage`. The included Editor script configures BaseColor, normal,
and packed metallic/smoothness maps, creates per-face materials, and writes
`room_pbr.prefab`. It supports URP Lit, HDRP Lit, or the built-in Standard
shader, selecting the first available option in that order. The same setup can
be rerun from **Tools > v3b_v3 > Apply PBR**. Imported assets are placed under
`Assets/v3b_v3`.

## Reproducibility and publication

By default CHORD is rerun for every new trace-back image. Optional caches are
accepted only when the complete input-image SHA-256 matches, and all five files
(`input`, BaseColor, normal, roughness, metallic) are present.

The source tree is intended to be GitHub-ready: generated outputs, checkpoints,
datasets, caches, machine-local configuration, and Unity packages are ignored.
Run `bash check_release.sh` before publishing to verify the frozen algorithm
hashes, script closure, source hygiene, and Unity archive builder.
No project license is selected in this package; the repository owner must add
the intended license before presenting it as open-source software.
