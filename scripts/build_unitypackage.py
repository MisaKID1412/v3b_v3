#!/usr/bin/env python3
"""Build and structurally verify an importable Unity ``.unitypackage`` archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import uuid
from pathlib import Path, PurePosixPath


REQUIRED_RELATIVE_PATHS = (
    "room.obj",
    "room.mtl",
    "pbr_textures/basecolor/floor.png",
    "pbr_textures/normal/floor.png",
    "pbr_textures/roughness/floor.png",
    "pbr_textures/metallic/floor.png",
    "pbr_textures/unity_metallic_smoothness/floor.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--asset-root", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_guid(asset_path: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"v3b_v3-unitypackage:{asset_path}").hex


def texture_meta(guid: str, asset_path: str) -> bytes:
    lower = asset_path.lower()
    is_normal = "/pbr_textures/normal/" in lower
    is_linear = is_normal or any(
        marker in lower
        for marker in (
            "/pbr_textures/roughness/",
            "/pbr_textures/metallic/",
            "/pbr_textures/unity_metallic_smoothness/",
        )
    )
    srgb = 0 if is_linear else 1
    texture_type = 1 if is_normal else 0
    return f"""fileFormatVersion: 2
guid: {guid}
TextureImporter:
  internalIDToNameTable: []
  externalObjects: {{}}
  serializedVersion: 12
  mipmaps:
    mipMapMode: 0
    enableMipMap: 1
    sRGBTexture: {srgb}
    linearTexture: 0
    fadeOut: 0
    borderMipMap: 0
    mipMapsPreserveCoverage: 0
    alphaTestReferenceValue: 0.5
    mipMapFadeDistanceStart: 1
    mipMapFadeDistanceEnd: 3
  bumpmap:
    convertToNormalMap: 0
    externalNormalMap: 0
    heightScale: 0.25
    normalMapFilter: 0
    flipGreenChannel: 0
  isReadable: 0
  streamingMipmaps: 0
  grayScaleToAlpha: 0
  generateCubemap: 6
  cubemapConvolution: 0
  seamlessCubemap: 0
  textureFormat: 1
  maxTextureSize: 16384
  textureSettings:
    serializedVersion: 2
    filterMode: 1
    aniso: 1
    mipBias: 0
    wrapU: 1
    wrapV: 1
    wrapW: 1
  nPOTScale: 0
  lightmap: 0
  compressionQuality: 50
  spriteMode: 0
  spriteExtrude: 1
  spriteMeshType: 1
  alignment: 0
  spritePivot: {{x: 0.5, y: 0.5}}
  spritePixelsToUnits: 100
  spriteBorder: {{x: 0, y: 0, z: 0, w: 0}}
  alphaUsage: 1
  alphaIsTransparency: 0
  spriteTessellationDetail: -1
  textureType: {texture_type}
  textureShape: 1
  singleChannelComponent: 0
  flipbookRows: 1
  flipbookColumns: 1
  maxTextureSizeSet: 0
  compressionQualitySet: 0
  textureFormatSet: 0
  ignorePngGamma: 0
  applyGammaDecoding: 0
  swizzle: 50462976
  cookieLightType: 0
  platformSettings: []
  spriteSheet:
    serializedVersion: 2
    sprites: []
    outline: []
    physicsShape: []
    bones: []
    spriteID:
    internalID: 0
    vertices: []
    indices:
    edges: []
    weights: []
    secondaryTextures: []
    nameFileIdTable: {{}}
  mipmapLimitGroupName:
  pSDRemoveMatte: 0
  userData:
  assetBundleName:
  assetBundleVariant:
""".encode("utf-8")


def model_meta(guid: str) -> bytes:
    return f"""fileFormatVersion: 2
guid: {guid}
ModelImporter:
  serializedVersion: 23
  internalIDToNameTable: []
  externalObjects: {{}}
  materials:
    materialImportMode: 1
    materialName: 0
    materialSearch: 1
    materialLocation: 1
  animations:
    legacyGenerateAnimations: 4
    bakeSimulation: 0
    resampleCurves: 1
    optimizeGameObjects: 0
    removeConstantScaleCurves: 0
    motionNodeName:
    rigImportErrors:
    rigImportWarnings:
    animationImportErrors:
    animationImportWarnings:
    animationRetargetingWarnings:
    animationDoRetargetingWarnings: 0
    importAnimatedCustomProperties: 0
    importConstraints: 0
    animationCompression: 1
    animationRotationError: 0.5
    animationPositionError: 0.5
    animationScaleError: 0.5
    animationWrapMode: 0
    extraExposedTransformPaths: []
    extraUserProperties: []
    clipAnimations: []
    isReadable: 0
  meshes:
    lODScreenPercentages: []
    globalScale: 1
    meshCompression: 0
    addColliders: 0
    useSRGBMaterialColor: 1
    sortHierarchyByName: 1
    importVisibility: 1
    importBlendShapes: 1
    importCameras: 1
    importLights: 1
    fileIdsGeneration: 2
    swapUVChannels: 0
    generateSecondaryUV: 0
    useFileUnits: 1
    keepQuads: 0
    weldVertices: 1
    bakeAxisConversion: 0
    preserveHierarchy: 0
    skinWeightsMode: 0
    maxBonesPerVertex: 4
    minBoneWeight: 0.001
    optimizeBones: 1
    meshOptimizationFlags: -1
    indexFormat: 0
    secondaryUVAngleDistortion: 8
    secondaryUVAreaDistortion: 15.000001
    secondaryUVHardAngle: 88
    secondaryUVMarginMethod: 1
    secondaryUVMinLightmapResolution: 40
    secondaryUVMinObjectScale: 1
    secondaryUVPackMargin: 4
    useFileScale: 1
  tangentSpace:
    normalSmoothAngle: 60
    normalImportMode: 0
    tangentImportMode: 3
    normalCalculationMode: 4
    legacyComputeAllNormalsFromSmoothingGroupsWhenMeshHasBlendShapes: 0
    blendShapeNormalImportMode: 1
    normalSmoothingSource: 0
  userData:
  assetBundleName:
  assetBundleVariant:
""".encode("utf-8")


def mono_meta(guid: str) -> bytes:
    return f"""fileFormatVersion: 2
guid: {guid}
MonoImporter:
  externalObjects: {{}}
  serializedVersion: 2
  defaultReferences: []
  executionOrder: 0
  icon: {{instanceID: 0}}
  userData:
  assetBundleName:
  assetBundleVariant:
""".encode("utf-8")


def text_meta(guid: str) -> bytes:
    return f"""fileFormatVersion: 2
guid: {guid}
TextScriptImporter:
  externalObjects: {{}}
  userData:
  assetBundleName:
  assetBundleVariant:
""".encode("utf-8")


def default_meta(guid: str) -> bytes:
    return f"""fileFormatVersion: 2
guid: {guid}
DefaultImporter:
  externalObjects: {{}}
  userData:
  assetBundleName:
  assetBundleVariant:
""".encode("utf-8")


def generated_meta(guid: str, source: Path, asset_path: str) -> bytes:
    suffix = source.suffix.lower()
    if suffix == ".png":
        return texture_meta(guid, asset_path)
    if suffix == ".obj":
        return model_meta(guid)
    if suffix == ".cs":
        return mono_meta(guid)
    if suffix in {".json", ".md", ".txt"}:
        return text_meta(guid)
    return default_meta(guid)


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def add_asset(archive: tarfile.TarFile, source: Path, asset_path: str) -> None:
    guid = deterministic_guid(asset_path)
    asset_info = archive.gettarinfo(str(source), arcname=f"{guid}/asset")
    asset_info.mtime = 0
    asset_info.uid = 0
    asset_info.gid = 0
    asset_info.uname = ""
    asset_info.gname = ""
    with source.open("rb") as stream:
        archive.addfile(asset_info, stream)

    sidecar_meta = source.with_name(source.name + ".meta")
    meta = sidecar_meta.read_bytes() if sidecar_meta.exists() else generated_meta(guid, source, asset_path)
    add_bytes(archive, f"{guid}/asset.meta", meta)
    add_bytes(archive, f"{guid}/pathname", asset_path.encode("utf-8"))


def verify_package(package: Path, asset_root: str, expected_count: int) -> dict[str, object]:
    with tarfile.open(package, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        groups: dict[str, set[str]] = {}
        pathnames: list[str] = []
        for name in members:
            parts = PurePosixPath(name).parts
            if len(parts) != 2:
                raise RuntimeError(f"invalid unitypackage member path: {name}")
            groups.setdefault(parts[0], set()).add(parts[1])
        for guid, names in groups.items():
            required = {"asset", "asset.meta", "pathname"}
            if not required.issubset(names):
                raise RuntimeError(f"incomplete unitypackage group {guid}: {sorted(names)}")
            pathname_stream = archive.extractfile(members[f"{guid}/pathname"])
            if pathname_stream is None:
                raise RuntimeError(f"missing pathname payload for {guid}")
            pathname = pathname_stream.read().decode("utf-8")
            if not pathname.startswith(asset_root + "/"):
                raise RuntimeError(f"asset escaped package root: {pathname}")
            if ".." in PurePosixPath(pathname).parts:
                raise RuntimeError(f"unsafe pathname: {pathname}")
            pathnames.append(pathname)

    if len(groups) != expected_count:
        raise RuntimeError(f"asset count mismatch: package={len(groups)} expected={expected_count}")
    if len(pathnames) != len(set(pathnames)):
        raise RuntimeError("duplicate Unity asset path")
    missing = [
        f"{asset_root}/{relative}"
        for relative in REQUIRED_RELATIVE_PATHS
        if f"{asset_root}/{relative}" not in pathnames
    ]
    if missing:
        raise RuntimeError(f"required Unity assets missing: {missing}")
    editor_scripts = [
        pathname
        for pathname in pathnames
        if pathname.startswith(f"{asset_root}/UnityImportSettings/Editor/")
        and pathname.endswith(".cs")
    ]
    if not editor_scripts:
        raise RuntimeError("Unity Editor setup script is missing")
    return {
        "status": "verified",
        "archive_format": "gzip-compressed Unity package tar",
        "asset_root": asset_root,
        "asset_count": len(groups),
        "member_count": len(groups) * 3,
        "required_assets_present": True,
        "editor_scripts": editor_scripts,
    }


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    asset_root = args.asset_root.strip("/")
    if not asset_root.startswith("Assets/") or ".." in PurePosixPath(asset_root).parts:
        raise ValueError("--asset-root must be a safe path below Assets/")
    if args.out_file.suffix != ".unitypackage":
        raise ValueError("--out-file must end with .unitypackage")
    if args.out_file.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_file}")

    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and not path.name.endswith(".meta")
        and path.suffix != ".unitypackage"
        and path.name != ".DS_Store"
    )
    relative_paths = {path.relative_to(source_dir).as_posix() for path in files}
    missing_source = [path for path in REQUIRED_RELATIVE_PATHS if path not in relative_paths]
    if missing_source:
        raise FileNotFoundError(f"source package is incomplete: {missing_source}")
    if not any(
        relative.startswith("UnityImportSettings/Editor/") and relative.endswith(".cs")
        for relative in relative_paths
    ):
        raise FileNotFoundError("source package has no Unity Editor setup script")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out_file, "w:gz", format=tarfile.GNU_FORMAT, compresslevel=6) as archive:
        for source in files:
            relative = source.relative_to(source_dir).as_posix()
            add_asset(archive, source, f"{asset_root}/{relative}")

    verification = verify_package(args.out_file, asset_root, len(files))
    verification.update(
        {
            "package": args.out_file.name,
            "bytes": args.out_file.stat().st_size,
            "sha256": file_sha256(args.out_file),
        }
    )
    manifest_path = args.out_file.with_suffix(args.out_file.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
