#!/usr/bin/env python3
"""Export a whole-territory v3b_v3 result as an honest Unity PBR package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


CHANNELS = ("basecolor", "normal", "roughness", "metallic")


def portable_manifest_value(value: object) -> object:
    """Remove machine-specific absolute paths from published Unity metadata."""
    if isinstance(value, dict):
        return {key: portable_manifest_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_manifest_value(item) for item in value]
    if isinstance(value, tuple):
        return [portable_manifest_value(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        is_absolute = normalized.startswith(("/", "~/")) or bool(
            re.match(r"^[A-Za-z]:/", normalized)
        )
        if is_absolute:
            return f"pipeline-artifact:{normalized.rstrip('/').rsplit('/', 1)[-1]}"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-unity-dir", type=Path, required=True)
    parser.add_argument("--pbr-run-dir", type=Path, required=True)
    parser.add_argument("--editor-script", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def source_metadata_path(run_dir: Path) -> Path:
    candidates = (
        run_dir / "metadata_adaptive_whole_territory_pbr.json",
        run_dir / "metadata_native_single_exemplar_convfield_pbr.json",
        run_dir / "metadata_newstart_test_territory_pbr.json",
        run_dir / "metadata_chord_atlas_overlap_pbr_v8.json",
        run_dir / "metadata_chord_atlas_overlap_pbr_v7.json",
        run_dir / "metadata_chord_atlas_overlap_pbr_v6.json",
        run_dir / "metadata_chord_atlas_overlap_pbr_v5.json",
        run_dir / "metadata_chord_atlas_overlap_pbr_v4.json",
        run_dir / "metadata_unified_globalfield_pbr.json",
        run_dir / "metadata_native_scale_multiview_pbr.json",
        run_dir / "metadata_pbr_placement.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no supported PBR metadata under {run_dir}")


def face_key(face: str) -> tuple[int, int, str]:
    if face == "floor":
        return (0, 0, face)
    if face == "ceiling":
        return (1, 0, face)
    match = re.fullmatch(r"wall_(\d+)", face)
    return (2, int(match.group(1)), face) if match else (3, 0, face)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_geometry(source: Path, destination: Path) -> None:
    for name in ("room.obj", "room.mtl", "structure_package_preview.png"):
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)
    require(destination / "room.obj")
    require(destination / "room.mtl")


def write_unity_metallic_smoothness(run_dir: Path, out_dir: Path, faces: list[str]) -> None:
    destination = out_dir / "pbr_textures" / "unity_metallic_smoothness"
    destination.mkdir(parents=True)
    for face in faces:
        metal = np.asarray(
            Image.open(run_dir / "pbr_textures" / "metallic" / f"{face}.png").convert("L"),
            dtype=np.uint8,
        )
        rough = np.asarray(
            Image.open(run_dir / "pbr_textures" / "roughness" / f"{face}.png").convert("L"),
            dtype=np.uint8,
        )
        if metal.shape != rough.shape:
            raise ValueError(f"PBR shape mismatch for {face}: {metal.shape} vs {rough.shape}")
        packed = np.empty((*metal.shape, 4), dtype=np.uint8)
        packed[..., :3] = metal[..., None]
        packed[..., 3] = 255 - rough
        Image.fromarray(packed, mode="RGBA").save(destination / f"{face}.png")


def main() -> int:
    args = parse_args()
    require(args.geometry_unity_dir)
    metadata_path = source_metadata_path(args.pbr_run_dir)
    require(args.editor_script)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    copy_geometry(args.geometry_unity_dir, args.out_dir)

    base_root = args.pbr_run_dir / "pbr_textures" / "basecolor"
    require(base_root)
    faces = sorted((path.stem for path in base_root.glob("*.png")), key=face_key)
    if not faces:
        raise RuntimeError(f"no PBR faces under {base_root}")
    sizes: dict[str, list[int]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for channel in CHANNELS:
        source = args.pbr_run_dir / "pbr_textures" / channel
        require(source)
        destination = args.out_dir / "pbr_textures" / channel
        shutil.copytree(source, destination)
    texture_dir = args.out_dir / "textures"
    texture_dir.mkdir()
    for face in faces:
        hashes[face] = {}
        expected_size = None
        for channel in CHANNELS:
            path = args.pbr_run_dir / "pbr_textures" / channel / f"{face}.png"
            require(path)
            with Image.open(path) as image:
                size = [int(image.width), int(image.height)]
            expected_size = size if expected_size is None else expected_size
            if size != expected_size:
                raise ValueError(f"channel size mismatch for {face}: {channel}={size}, expected={expected_size}")
            hashes[face][channel] = sha256(path)
        sizes[face] = expected_size
        shutil.copy2(
            args.pbr_run_dir / "pbr_textures" / "basecolor" / f"{face}.png",
            texture_dir / f"{face}.png",
        )
    write_unity_metallic_smoothness(args.pbr_run_dir, args.out_dir, faces)

    editor_destination = args.out_dir / "UnityImportSettings" / "Editor" / args.editor_script.name
    editor_destination.parent.mkdir(parents=True)
    shutil.copy2(args.editor_script, editor_destination)
    preview_names = (
        "adaptive_whole_territory_pbr_overview.jpg",
        "native_convfield_vs_v1.jpg",
        "scale_locked_wholefield_overview.jpg",
        "basecolor_contact_sheet.png",
        "continuous_scale_locked_nontile_pbr_v8_overview.jpg",
        "unified_lattice_free_globalfield_pbr_overview.jpg",
        "native_scale_multiview_pbr_overview.jpg",
        "aligned_scale_locked_pbr_overview.jpg",
    )
    for preview_name in preview_names:
        preview_source = args.pbr_run_dir / "previews" / preview_name
        if preview_source.exists():
            (args.out_dir / "previews").mkdir(exist_ok=True)
            shutil.copy2(preview_source, args.out_dir / "previews" / preview_source.name)
            break

    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adaptive_whole = metadata_path.name == "metadata_adaptive_whole_territory_pbr.json"
    generalized_v1 = source_metadata.get("material_routing") == "generalized"
    native_convfield = metadata_path.name == "metadata_native_single_exemplar_convfield_pbr.json"
    newstart_test = metadata_path.name == "metadata_newstart_test_territory_pbr.json"
    continuous_v8 = metadata_path.name == "metadata_chord_atlas_overlap_pbr_v8.json"
    continuous_v7 = metadata_path.name == "metadata_chord_atlas_overlap_pbr_v7.json"
    continuous_v6 = metadata_path.name == "metadata_chord_atlas_overlap_pbr_v6.json"
    continuous_v5 = metadata_path.name == "metadata_chord_atlas_overlap_pbr_v5.json"
    conditional_v4 = metadata_path.name == "metadata_chord_atlas_overlap_pbr_v4.json"
    unified_v3 = metadata_path.name == "metadata_unified_globalfield_pbr.json"
    native_v2 = metadata_path.name == "metadata_native_scale_multiview_pbr.json"
    continuous_atlas = continuous_v8 or continuous_v7 or continuous_v6 or continuous_v5 or conditional_v4
    native_multiview = continuous_atlas or unified_v3 or native_v2 or newstart_test or native_convfield
    continuous_field_metadata = None
    if (continuous_v8 or continuous_v7 or continuous_v6 or continuous_v5) and source_metadata.get("input_dir"):
        candidate = (
            Path(source_metadata["input_dir"]).parent
            / "metadata_scale_locked_continuous_field_v5.json"
        )
        if candidate.exists():
            continuous_field_metadata = str(candidate)
    manifest = {
        "method": (
            "v3b_v3_adaptive_whole_territory_scale_locked_unity_pbr"
            if adaptive_whole
            else "v3b_newStart_generalized_material_routing_scale_locked_unity_pbr_v1"
            if generalized_v1
            else "v3b_newStart_native64_single_exemplar_convfield_unity_pbr"
            if native_convfield
            else "v3b_newStart_test_native64_joint_global_territory_unity_pbr"
            if newstart_test
            else (
                "native_multiview_scale_calibrated_continuous_material_transfer_unity_pbr_v8"
                if continuous_v8
                else (
                    "native_multiview_continuous_material_transfer_unity_pbr_v7"
                    if continuous_v7
                    else (
                        "native_multiview_scale_locked_continuous_material_field_unity_pbr_v6"
                        if continuous_v6 or continuous_v5
                        else (
                            "native_multiview_conditioned_continuous_atlas_unity_pbr_v4"
                            if conditional_v4
                            else (
                                "unified_lattice_free_native_scale_globalfield_unity_pbr_v3"
                                if unified_v3
                                else (
                                    "native_scale_traceback_multiview_wholefield_unity_pbr_v2"
                                    if native_v2
                                    else "matseg_traceback_chord_shared_mapping_scale_locked_wholefield_unity_pbr_v1"
                                )
                            )
                        )
                    )
                )
            )
        ),
        "version_name": (
            "v3b_v3_same_path_adaptive_whole_territory_full_pbr"
            if adaptive_whole
            else "v3b_newStart_generalized_v1_test_full_pbr"
            if generalized_v1
            else "v3b_newStart_test_native_single_exemplar_continuous_full_pbr"
            if native_convfield
            else "v3b_newStart_test_fullroom_native64_nontile_full_pbr"
            if newstart_test
            else (
                "v3b_newStart_scale_locked_continuous_material_transfer_full_pbr_v8"
                if continuous_v8
                else (
                    "v3b_newStart_scale_locked_continuous_material_transfer_full_pbr_v7"
                    if continuous_v7
                    else (
                        "v3b_newStart_scale_locked_continuous_material_field_full_pbr_v6"
                        if continuous_v6 or continuous_v5
                        else (
                            "v3b_newStart_native_multiview_conditional_globalfield_full_pbr_v4"
                            if conditional_v4
                            else (
                                "v3b_newStart_unified_lattice_free_globalfield_full_pbr_v3"
                                if unified_v3
                                else (
                                    "v3b_newStart_native_scale_multiview_full_pbr_v2"
                                    if native_v2
                                    else "v3b_newStart_aligned_wholefield_full_pbr"
                                )
                            )
                        )
                    )
                )
            )
        ),
        "pbr_enabled_unity_package": True,
        "entry_obj": "room.obj",
        "entry_prefab_after_unity_import": "room_pbr.prefab",
        "geometry_source": str(args.geometry_unity_dir),
        "pbr_run_source": str(args.pbr_run_dir),
        "source_metadata": str(metadata_path),
        "continuous_field_metadata": continuous_field_metadata,
        "source_chord_outputs": (
            {
                "mode": "same-size full CHORD outputs drive one whole-territory field",
                "per_window_texture_assets": False,
                "upstream_contract": source_metadata.get("upstream_contract"),
            }
            if adaptive_whole
            else source_metadata.get("chord_output_dir")
            if native_convfield
            else source_metadata.get("chord_reference_dir")
            if newstart_test
            else
            {
                "mode": (
                    "calibration responses only; continuous material transfer is evaluated once over the whole atlas"
                    if continuous_v8
                    else "same-coordinate overlapping continuous-atlas inference"
                ),
                "per_window_texture_assets": False,
            }
            if continuous_atlas
            else {
                "anchor": source_metadata.get("anchor_chord_dir"),
                "supplemental": source_metadata.get("supplemental_chord_dir"),
            }
            if unified_v3 or native_v2
            else source_metadata.get("source_full_chord_outputs")
        ),
        "source_chord_input_metadata": (
            (source_metadata.get("upstream_contract") or {}).get("candidate_metadata")
            if adaptive_whole
            else source_metadata.get("trace_metadata")
            if native_convfield
            else source_metadata.get("trace_report")
            if newstart_test
            else None
            if continuous_atlas
            else source_metadata.get("native_trace_metadata")
            if unified_v3 or native_v2
            else source_metadata.get("source_chord_input_metadata")
        ),
        "chord_contract": (
            (
                "highest-weight reverse-projected and rectified material observation -> "
                "normalized 512x512 CHORD input -> same-size aligned basecolor, normal, "
                "roughness, and metallic; the complete output represents the traced native "
                "support and is mapped back with its recorded physical scale"
            )
            if adaptive_whole
            else
            (
                "highest reverse-projection-weight raw contributor -> exact native 64x64 "
                "rectified crop -> frozen CHORD 64x64 basecolor+normal+roughness+metallic; "
                "no input resize and no output resize"
            )
            if native_convfield or newstart_test
            else
            (
                "CHORD windows provide per-material calibration responses only; crop-dependent "
                "baselines are discarded and the learned response is evaluated continuously "
                "over native atlas features"
                if continuous_v8
                else (
                    "continuous native atlas -> overlapping same-coordinate CHORD inference; "
                    "windows are memory partitions and are never placed as texture patches"
                )
            )
            if continuous_atlas
            else "trace-back patch, full CHORD basecolor+normal+roughness+metallic"
            if not native_multiview
            else "native rectified inner crop -> same-size CHORD basecolor+normal+roughness+metallic; no post-CHORD recrop"
        ),
        "synthesis_contract": (
            source_metadata.get("generalization_contract")
            if adaptive_whole
            else source_metadata.get("contract")
            if native_multiview
            else source_metadata.get("channel_contract")
        ),
        "basecolor_is_direct_fullface_chord": False,
        "basecolor_description": (
            (
                "one scale-locked whole-territory realization per material; route chosen "
                "from measured texture statistics, with no fixed tile grid, patch placement, "
                "quilt, or face/material-name exception"
            )
            if adaptive_whole
            else
            (
                "accepted v1 low/high-frequency material synthesis with smooth, stochastic, "
                "or structured route selected from exemplar statistics rather than face type; "
                "all PBR channels retain the accepted v1 shared spatial mapping"
            )
            if generalized_v1
            else
            (
                "one fully convolutional single-exemplar material field evaluated once for "
                "each connected structured territory; trained and evaluated on the native "
                "atlas pixel grid from the exact CHORD 64x64 PBR reference, with no tile, "
                "patch placement, quilt, graph cut, random Fourier phase, or resize"
            )
            if native_convfield
            else
            (
                "one joint full-domain spectral field generated once per connected MatSeg "
                "material territory from the frozen CHORD native-64 PBR reference; frequencies "
                "remain cycles per native atlas pixel, wavelengths unsupported by the reference "
                "are suppressed, and no patch, tile, quilt, graph cut, mirror repeat, or resize "
                "is used"
            )
            if newstart_test
            else
            (
                "one continuous material-wide field per structured territory; native scale, "
                "anisotropy, and RGB marginal statistics measured from all MatSeg-compatible "
                "robust multi-view observations; no source patch, tile, quilt, mirror repeat, "
                "or Fourier phase retrieval"
            )
            if continuous_v8 or continuous_v7 or continuous_v6 or continuous_v5
            else (
                (
                    "robust raw multi-view observations fused at native atlas coordinates; "
                    "missing texels completed as an all-observation-conditioned global field; "
                    "no source patch, tile, quilt, mirror repeat, or random Fourier phase"
                )
                if conditional_v4
                else (
                    "one continuous lattice-free material field per territory, using a global low-band guide and native-scale multiview spectral detail; no patch placement"
                    if unified_v3
                    else (
                        "native-scale whole-material field synthesized from one highest-weight anchor and same-material supplemental views"
                        if native_v2
                        else "scale-locked whole-material field synthesized from traced CHORD patch"
                    )
                )
            )
        ),
        "tileable_preprocessing_used": False,
        "faces": faces,
        "texture_sizes_wh": sizes,
        "texture_sha256": hashes,
        "unity_textures": {
            "basecolor_srgb": "textures/<face>.png",
            "normal_linear_normalmap": "pbr_textures/normal/<face>.png",
            "metallic_smoothness_linear": "pbr_textures/unity_metallic_smoothness/<face>.png",
            "source_roughness_linear": "pbr_textures/roughness/<face>.png",
            "source_metallic_linear": "pbr_textures/metallic/<face>.png",
        },
    }
    manifest = portable_manifest_value(manifest)
    (args.out_dir / "unity_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if adaptive_whole:
        readme = """# v3b_v3 Whole-Territory Unity PBR

This package contains the room mesh plus aligned BaseColor, tangent-space
normal, roughness, metallic, and Unity metallic/smoothness textures. Material
identity comes from MatSeg; every CHORD input is selected by the same
highest-weight reverse-projection trace-back rule and processed at its saved
512x512 size.

Each material territory is synthesized once as a scale-locked whole-domain
field. The route is selected from measured texture evidence, not from a room,
face, or material name. No fixed tile grid, patch pasting, or quilting is used.
All PBR channels share the same spatial mapping and hard material boundaries.

Copy this folder under a Unity project's `Assets`, then use the included Editor
importer to create Lit materials and `room_pbr.prefab`. BaseColor is sRGB;
normal and scalar PBR data are imported as linear textures. The packed Unity
texture stores metallic in RGB and smoothness (`1 - roughness`) in alpha.
"""
    elif generalized_v1:
        readme = """# v3b_newStart Generalized v1 Unity PBR Test

This build preserves the accepted v1 synthesis mechanisms, 2x atlas resolution,
shared PBR spatial mappings, material boundaries, and source trace-back/CHORD
assets. The only synthesis-routing change is that smooth, stochastic, or
structured processing is selected from the material exemplar's high-frequency,
edge, periodicity, and directional statistics. Face type (floor, wall, or
ceiling) is not used for route selection or structured-material prompting.

Unity uses the packed metallic/smoothness texture (RGB = metallic, A =
1-roughness). BaseColor is sRGB; every other map is imported as linear data.
Use the included Editor importer to create Lit materials and room_pbr.prefab.
"""
    elif native_convfield:
        readme = """# v3b_newStart Native Single-Exemplar Unity PBR Test

This is the v3b_newStart test build for the L-shaped room. MatSeg is used only
for material identity and structured territory grouping. Original v3b geometry,
projection, depth, visibility, and object-rejection logic is unchanged. Each
material reference is the exact native 64x64 rectified crop from the highest
reverse-projection-weight valid contributor. The frozen CHORD model receives
64x64 and returns aligned 64x64 BaseColor, normal, roughness, and metallic maps;
neither input nor output is resized.

A single-exemplar fully convolutional joint-PBR generator is trained on that
exact native pixel grid, then evaluated once for each connected material
territory at its atlas dimensions. It does not place, repeat, enlarge, quilt,
mirror, or graph-cut the reference patch. All PBR channels share the same
latent field and atlas coordinates, and structured material boundaries are
applied identically to every channel.

Unity uses the packed metallic/smoothness texture (RGB = metallic, A =
1-roughness). BaseColor is sRGB; every other map is imported as linear data.
Use the included Editor importer to create Lit materials and room_pbr.prefab.
"""
    elif newstart_test:
        readme = """# v3b_newStart Test Full-Room Unity PBR

This is the v3b_newStart test build for the L-shaped room. MatSeg is used only
for material identity and territory grouping. Every material reference is an
exact native 64x64 crop traced from the highest reverse-projection-weight raw
contributor and processed by the frozen CHORD weights at 64x64. Neither the
CHORD input nor its output is resized.

Each connected material territory is generated once as one joint full-domain
PBR field. Frequencies are measured in cycles per native atlas pixel and no
wavelength larger than the evidence in the 64x64 reference is invented. There
is no patch placement, tile, quilt, graph cut, mirror repeat, or per-channel
spatial synthesis. BaseColor, normal, roughness, and metallic share one field.

Unity uses the packed metallic/smoothness texture (RGB = metallic, A =
1-roughness). BaseColor is sRGB; every other map is imported as linear data.
Use the included Editor importer to create Lit materials and room_pbr.prefab.
"""
    elif continuous_v8:
        readme = """# v3b_newStart v8 Unity Full PBR

This package contains one continuous scale-locked BaseColor field and aligned
tangent-space normal, roughness, and metallic maps for every room face. MatSeg
is used only for material identity; the original v3b projection, visibility,
depth, object rejection, and atlas coordinates remain unchanged.

No CHORD window is copied, placed, tiled, quilted, mirrored, or resized into the
final textures. CHORD responses are used to learn one material-specific PBR
transfer. Crop-dependent PBR baselines are discarded, response strength is
calibrated from all same-material predictions, and the transfer is evaluated
once over continuous native-atlas features. BaseColor is one whole material
field whose scale and directional statistics come from robust MatSeg-compatible
multi-view observations at native atlas scale.

Unity uses the packed metallic/smoothness texture (RGB = metallic, A =
1-roughness). BaseColor is sRGB; every other map is imported as linear data.
After copying this folder under a Unity project's `Assets`, use the included
Editor importer to create Lit materials and `room_pbr.prefab`.

The JSON manifest and source audit record the scale, continuity, alignment, and
non-periodic contracts for all eight faces.
"""
    elif continuous_atlas:
        readme = """# v3b_newStart Continuous Atlas Unity Full PBR

This package contains one continuous scale-locked BaseColor field and aligned
tangent-space normal, roughness, and metallic maps for every room face. MatSeg
is used only for material identity. BaseColor is synthesized once per complete
material territory from multi-view statistics measured at native atlas scale;
there is no source patch, repeated tile, quilt, or mirror placement. CHORD is
evaluated in overlapping same-coordinate windows only for intrinsic low colour
and aligned PBR prediction; no inference window becomes a texture asset.

Unity uses the packed metallic/smoothness texture (RGB = metallic, A =
1-roughness). BaseColor is sRGB; every other map is imported as linear data.
After copying this folder under a Unity project's `Assets`, use the included
Editor importer to create Lit materials and `room_pbr.prefab`.

The JSON manifest records the continuous-field and coordinate-preserving PBR
contracts. No tile, quilt, mirror repeat, patch placement, or per-channel
texture synthesis is used.
"""
    else:
        readme = """# v3b_newStart Unity Full PBR

This package contains the L-room geometry and four aligned texture channels for
each face: BaseColor, tangent-space normal, roughness, and metallic. Unity uses
the additional packed metallic/smoothness texture (RGB = metallic, A =
1-roughness). BaseColor is sRGB; every other map is imported as linear data.

After copying this folder under a Unity project's `Assets`, the included Editor
script creates Lit materials and `room_pbr.prefab`. If automatic setup does not
run, use `Tools > v3b_v3 > Apply PBR`.

The JSON manifest records the exact trace-back CHORD and whole-field sources.
All PBR channels share the same spatial synthesis mapping. No tileable
preprocessing or independent per-channel quilting is used.
"""
    (args.out_dir / "README_PBR_UNITY.md").write_text(readme, encoding="utf-8")
    print(f"[unity-pbr] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
