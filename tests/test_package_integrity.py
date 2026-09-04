from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_runner_has_the_complete_accepted_path() -> None:
    runner = ROOT / "run_from_images.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    text = runner.read_text(encoding="utf-8")
    stages = (
        "frontend_reconstruction",
        "unified_proposals",
        "material_identity_traceback",
        "chord_pbr",
        "material_layout",
        "territory_pbr",
        "unity_project",
        "unitypackage",
    )
    assert all(stage in text for stage in stages)
    assert "run_unified_frontend_v3.py" in text
    assert "--size 0" in text
    assert "--chord-output-support-mode full_normalized" in text
    assert "--no-use-discovered-material-masks" in text
    assert "build_unitypackage.py" in text
    assert "V3B_CALLER_RESUME_SET" in text
    assert "V3B_CALLER_RUN_FROM_SET" in text
    assert "V3B_CALLER_RUN_UNTIL_SET" in text


def test_accepted_backend_implementations_are_frozen() -> None:
    expected = {
        "scripts/frontend_v3/prepare_unified_region_proposals.py":
            "34a4245e5baa99c7f906b701c9d1cc586591e4e94e51dee0b12f1fe04e85fc48",
        "scripts/frontend_v3/filter_incoherent_material_groups.py":
            "cbe078d51d1157ca871dfc9ca3f007d8dc8071e2148471b1cc0b0c7f65a3831d",
        "scripts/frontend_v3/run_matseg_floor_diagnostic.py":
            "9a0beedb8e2060960922d10a0ed2ce51e5e5c4037fdb3593468dfe3bc5e51e59",
        "scripts/frontend_v3/run_matseg_material_identity_from_v3b_regions.py":
            "0e2de6886b1c6b922190f07f7bf293b0c95ed0d2ae9948765102e3b803b057f7",
        "scripts/frontend_v3/run_matseg_material_identity_contact.py":
            "166e8bd2d9a2a2abd5f4b0219fd3f71dd83bc795e62387f59f548701734d309c",
        "scripts/frontend_v3/matseg_identity_resolver.py":
            "3ddc4ba802d6ea276826d80d7f48351025767876989b0557b83bd479a321156d",
        "scripts/frontend_v3/resolve_original_view_matseg_identity.py":
            "a6a5549ff7a78b6d8cd1c0fb6fada3dfd2bc77c0683f9861b0bc65b91cb58bdb",
        "scripts/frontend_v3/recover_untraceable_structural_support_by_matseg.py":
            "27cffb51794a027208023b701bd88990338cbb1ff413d767cf7e611bb7191105",
        "scripts/frontend_v3/regenerate_matseg_material_chord_inputs_by_traceback.py":
            "407f234cb240c9b67323e8143e955bb610f2c30dbff09f333140615cab76ea37",
        "scripts/compose_material_base_atlas_v1.py":
            "3ecb39a69e2c06e5b1080223ac17ca0e4c3c6d52b3226984b94147e703311199",
        "scripts/synthesize_adaptive_whole_territory_pbr_generalized_mirror_guard.py":
            "a7f597dc5f24f9e13988e5a9d315bac41ad30f7a5876c6bfeb5062a575bc4df9",
        "scripts/frontend_v3/generate_chord_view_contributor_region_priors.py":
            "3637803b094979175d43af34065cb3843ea25e40a792b92300f663aa3db3ee96",
        "scripts/refine_structured_material_territories_v2.py":
            "3f69d0c59faec138b75a999b8b3160531a138336b25b9560bc8c23f3fd10acce",
    }
    assert {name: digest(ROOT / name) for name in expected} == expected

    original_view_identity = (
        ROOT / "scripts/frontend_v3/run_matseg_material_identity_from_v3b_regions.py"
    ).read_text(encoding="utf-8")
    assert "matseg_region_material_identity_threshold_v1" in original_view_identity
    assert "similarity-threshold" in original_view_identity
    assert "resolve_material_ids" not in original_view_identity


def test_runtime_python_syntax_and_called_script_closure() -> None:
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    runner = (ROOT / "run_from_images.sh").read_text(encoding="utf-8")
    called = set(re.findall(r'\$PROJECT_DIR/(scripts/[A-Za-z0-9_./-]+\.(?:py|sh))', runner))
    assert called
    assert all((ROOT / name).is_file() for name in called), sorted(
        name for name in called if not (ROOT / name).is_file()
    )
    internal = ROOT / "restart_fullface_native_pipeline/scripts/run_new_dataset_full_pipeline_2080.sh"
    subprocess.run(["bash", "-n", str(internal)], check=True)
    internal_text = internal.read_text(encoding="utf-8")
    internal_called = set(
        re.findall(r'(?<![A-Za-z0-9_./-])(scripts/[A-Za-z0-9_./-]+\.py)', internal_text)
    )
    assert internal_called
    assert all((ROOT / name).is_file() for name in internal_called), sorted(
        name for name in internal_called if not (ROOT / name).is_file()
    )


def test_adaptive_pbr_audits_match_the_published_output_layout() -> None:
    scale_audit = (ROOT / "scripts/audit_scale_and_lattice.py").read_text(encoding="utf-8")
    wholefield_audit = (ROOT / "scripts/audit_wholefield_pbr.py").read_text(encoding="utf-8")
    runner = (ROOT / "run_from_images.sh").read_text(encoding="utf-8")
    assert "source_patches_scale_locked" in scale_audit
    assert "adaptive_material_field" in scale_audit
    assert "derived_from_adaptive_per_material_contracts" in wholefield_audit
    assert "contains_fixed_tile_grid" in wholefield_audit
    assert "reuse completed whole-territory PBR synthesis" in runner


def test_no_room_specific_route_or_private_machine_data() -> None:
    forbidden = (
        "/" + "mnt" + "/hdd/",
        "/" + "home" + "/jycni/",
        "166" + ".104.",
        "mrlab" + "@",
        "scene13_room150",
        "scene75_room249",
    )
    ignored = {
        Path(__file__).resolve(),
        (ROOT / "config/v3b.env").resolve(),
    }
    secret_patterns = (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    )
    for path in ROOT.rglob("*"):
        is_local_config = (
            path.parent.resolve() == (ROOT / "config").resolve()
            and path.name.startswith("local")
            and path.suffix == ".env"
        )
        if path.resolve() in ignored or is_local_config or not path.is_file():
            continue
        if any(part in {"outputs", ".cache", "tmp", "models", "checkpoints"} for part in path.parts):
            continue
        if path.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(token in text for token in forbidden), path
        assert not any(pattern.search(text) for pattern in secret_patterns), path


def test_publication_excludes_generated_and_large_binary_content() -> None:
    forbidden_suffixes = {".safetensors", ".pth", ".pt", ".ckpt", ".unitypackage"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {"outputs", ".cache", "tmp", "models", "checkpoints"} for part in path.parts):
            continue
        assert path.suffix.lower() not in forbidden_suffixes, path
        assert path.stat().st_size < 10 * 1024 * 1024, path


def test_unitypackage_builder_emits_a_structurally_complete_archive() -> None:
    required = (
        "room.obj",
        "room.mtl",
        "pbr_textures/basecolor/floor.png",
        "pbr_textures/normal/floor.png",
        "pbr_textures/roughness/floor.png",
        "pbr_textures/metallic/floor.png",
        "pbr_textures/unity_metallic_smoothness/floor.png",
        "UnityImportSettings/Editor/V3bV3PBRSetup.cs",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        for relative in required:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test-asset\n")
        package = root / "v3b_v3.unitypackage"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_unitypackage.py"),
                "--source-dir",
                str(source),
                "--out-file",
                str(package),
                "--asset-root",
                "Assets/v3b_v3",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        manifest = json.loads(
            package.with_suffix(".unitypackage.manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["status"] == "verified"
        assert manifest["required_assets_present"] is True
        assert manifest["asset_count"] == len(required)
        assert manifest["asset_root"] == "Assets/v3b_v3"
        assert manifest["editor_scripts"] == [
            "Assets/v3b_v3/UnityImportSettings/Editor/V3bV3PBRSetup.cs"
        ]
        assert manifest["package"] == "v3b_v3.unitypackage"
        assert str(root) not in json.dumps(manifest)


def test_published_manifests_use_portable_paths() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pbr = root / "pbr_full_normalized"
        unity = root / "unity_project"
        pbr.mkdir()
        unity.mkdir()

        exporter_source = (ROOT / "scripts/export_unity_aligned_pbr.py").read_text(
            encoding="utf-8"
        )
        exporter_tree = ast.parse(exporter_source)
        portable_function = next(
            node
            for node in exporter_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "portable_manifest_value"
        )
        namespace = {"re": re}
        exec(compile(ast.Module(body=[portable_function], type_ignores=[]), "<portable>", "exec"), namespace)
        sample = {
            "candidate": str(root / "private" / "candidate.json"),
            "nested": [str(root / "private" / "source.json")],
            "hash": "abc",
        }
        sanitized = namespace["portable_manifest_value"](sample)
        assert str(root) not in json.dumps(sanitized)
        assert sanitized["candidate"] == "pipeline-artifact:candidate.json"

        (pbr / "metadata_adaptive_whole_territory_pbr.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (unity / "unity_export_manifest.json").write_text("{}", encoding="utf-8")

        image_dir = root / "input_images"
        image_dir.mkdir()
        (image_dir / "000.png").write_bytes(b"image")
        identity = root / "identity.json"
        trace = root / "trace.json"
        identity.write_text("{}", encoding="utf-8")
        trace.write_text("{}", encoding="utf-8")
        layout = root / "layout"
        layout.mkdir()
        (layout / "metadata_material_placement.json").write_text("{}", encoding="utf-8")
        package = root / "v3b_v3.unitypackage"
        package.write_bytes(b"package")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/write_pipeline_manifest.py"),
                "--run-root",
                str(root),
                "--image-dir",
                str(image_dir),
                "--identity-contract",
                str(identity),
                "--trace-log",
                str(trace),
                "--layout-dir",
                str(layout),
                "--pbr-dir",
                str(pbr),
                "--unity-dir",
                str(unity),
                "--unitypackage",
                str(package),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        receipt_text = (root / "v3b_v3_end_to_end_manifest.json").read_text(
            encoding="utf-8"
        )
        assert str(root) not in receipt_text
        receipt = json.loads(receipt_text)
        assert receipt["input"]["image_dir"] == "input_images"
        assert receipt["artifacts"]["unitypackage"]["path"] == "v3b_v3.unitypackage"


def test_public_project_name_is_consistent() -> None:
    retired_name = "real" + "scenenpc"
    ignored_directories = {"outputs", ".cache", "tmp", "models", "checkpoints", "__pycache__", ".git"}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in ignored_directories for part in relative.parts):
            continue
        if not path.is_file() or path.suffix == ".pyc":
            continue
        if relative.as_posix() == "config/v3b.env" or (
            relative.parent.as_posix() == "config" and path.name.startswith("local")
        ):
            continue
        assert retired_name not in relative.as_posix().lower(), relative
        assert retired_name not in path.read_text(encoding="utf-8", errors="ignore").lower(), relative
    runner = (ROOT / "run_from_images.sh").read_text(encoding="utf-8")
    assert "--asset-root Assets/v3b_v3" in runner
    importer_path = ROOT / "unity/Editor/V3bV3PBRSetup.cs"
    assert "unity/Editor/V3bV3PBRSetup.cs" in runner
    importer = importer_path.read_text(encoding="utf-8")
    assert "namespace V3bV3.PBR" in importer
    assert "class V3bV3PBRSetup" in importer
    assert '[MenuItem("Tools/v3b_v3/Apply PBR")]' in importer


class PackageIntegrityTests(unittest.TestCase):
    def test_project_name(self) -> None:
        test_public_project_name_is_consistent()

    def test_adaptive_pbr_audits(self) -> None:
        test_adaptive_pbr_audits_match_the_published_output_layout()

    def test_public_runner(self) -> None:
        test_public_runner_has_the_complete_accepted_path()

    def test_frozen_backend(self) -> None:
        test_accepted_backend_implementations_are_frozen()

    def test_runtime_closure(self) -> None:
        test_runtime_python_syntax_and_called_script_closure()

    def test_no_private_data(self) -> None:
        test_no_room_specific_route_or_private_machine_data()

    def test_no_large_binaries(self) -> None:
        test_publication_excludes_generated_and_large_binary_content()

    def test_unitypackage_builder(self) -> None:
        test_unitypackage_builder_emits_a_structurally_complete_archive()

    def test_portable_manifests(self) -> None:
        test_published_manifests_use_portable_paths()


if __name__ == "__main__":
    unittest.main()
