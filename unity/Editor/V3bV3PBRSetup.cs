#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace V3bV3.PBR
{
    public sealed class V3bV3PBRSetup : AssetPostprocessor
    {
        private const string PackageMarker = "v3b_v3";

        [InitializeOnLoadMethod]
        private static void AutoSetupAfterImport()
        {
            EditorApplication.delayCall += () => SetupAll(false);
        }

        [MenuItem("Tools/v3b_v3/Apply PBR")]
        public static void ManualSetup()
        {
            SetupAll(true);
        }

        private void OnPreprocessTexture()
        {
            if (!IsInPackage(assetPath))
            {
                return;
            }

            var textureImporter = (TextureImporter)assetImporter;
            var path = Normalize(assetPath);
            textureImporter.mipmapEnabled = true;
            textureImporter.maxTextureSize = 16384;
            textureImporter.textureCompression = TextureImporterCompression.Uncompressed;

            if (path.Contains("/pbr_textures/normal/"))
            {
                textureImporter.textureType = TextureImporterType.NormalMap;
                textureImporter.sRGBTexture = false;
            }
            else if (path.Contains("/pbr_textures/unity_metallic_smoothness/") ||
                     path.Contains("/pbr_textures/metallic/") ||
                     path.Contains("/pbr_textures/roughness/"))
            {
                textureImporter.textureType = TextureImporterType.Default;
                textureImporter.sRGBTexture = false;
                textureImporter.alphaSource = TextureImporterAlphaSource.FromInput;
            }
            else
            {
                textureImporter.textureType = TextureImporterType.Default;
                textureImporter.sRGBTexture = true;
            }
        }

        private void OnPostprocessModel(GameObject model)
        {
            if (!IsRoomObj(assetPath))
            {
                return;
            }

            var folder = Normalize(Path.GetDirectoryName(assetPath) ?? string.Empty);
            EnsureMaterials(folder);
            AssignMaterials(model, folder);
        }

        private static void SetupAll(bool showDialog)
        {
            var roomPaths = AssetDatabase.GetAllAssetPaths()
                .Where(path => IsRoomObj(path))
                .ToArray();

            foreach (var roomPath in roomPaths)
            {
                var folder = Normalize(Path.GetDirectoryName(roomPath) ?? string.Empty);
                ReimportPbrTextures(folder);
                EnsureMaterials(folder);
                AssetDatabase.ImportAsset(roomPath, ImportAssetOptions.ForceUpdate);
                BuildPrefab(folder, roomPath);
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            if (showDialog)
            {
                EditorUtility.DisplayDialog(
                    "v3b_v3 PBR",
                    roomPaths.Length == 0
                        ? "No v3b_v3 room.obj was found under Assets."
                        : $"Updated PBR materials for {roomPaths.Length} v3b_v3 room asset(s).",
                    "OK");
            }
        }

        private static void ReimportPbrTextures(string folder)
        {
            var searchFolders = new[]
            {
                $"{folder}/textures",
                $"{folder}/pbr_textures/basecolor",
                $"{folder}/pbr_textures/normal",
                $"{folder}/pbr_textures/roughness",
                $"{folder}/pbr_textures/metallic",
                $"{folder}/pbr_textures/unity_metallic_smoothness"
            }.Where(AssetDatabase.IsValidFolder).ToArray();

            if (searchFolders.Length == 0)
            {
                return;
            }

            foreach (var texturePath in AssetDatabase.FindAssets("t:Texture2D", searchFolders)
                         .Select(AssetDatabase.GUIDToAssetPath)
                         .Distinct(StringComparer.OrdinalIgnoreCase))
            {
                AssetDatabase.ImportAsset(texturePath, ImportAssetOptions.ForceUpdate);
            }
        }

        private static void EnsureMaterials(string folder)
        {
            var faces = DiscoverFaces(folder);
            if (faces.Length == 0)
            {
                Debug.LogWarning($"v3b_v3 PBR: no face textures found under {folder}/textures.");
                return;
            }
            var materialFolder = $"{folder}/Materials";
            if (!AssetDatabase.IsValidFolder(materialFolder))
            {
                AssetDatabase.CreateFolder(folder, "Materials");
            }

            var shader = Shader.Find("Universal Render Pipeline/Lit") ??
                         Shader.Find("HDRP/Lit") ??
                         Shader.Find("Standard");
            if (shader == null)
            {
                Debug.LogWarning("v3b_v3 PBR: no Lit/Standard shader found.");
                return;
            }

            foreach (var face in faces)
            {
                var materialPath = $"{materialFolder}/{face}.mat";
                var material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
                if (material == null)
                {
                    material = new Material(shader) { name = face };
                    AssetDatabase.CreateAsset(material, materialPath);
                }
                else if (material.shader != shader)
                {
                    material.shader = shader;
                }

                ConfigureMaterial(material, folder, face);
                EditorUtility.SetDirty(material);
            }
        }

        private static void ConfigureMaterial(Material material, string folder, string face)
        {
            var baseColor = LoadTexture($"{folder}/textures/{face}.png");
            var normal = LoadTexture($"{folder}/pbr_textures/normal/{face}.png");
            var metallicSmoothness = LoadTexture($"{folder}/pbr_textures/unity_metallic_smoothness/{face}.png");

            SetTextureIfPresent(material, "_MainTex", baseColor);
            SetTextureIfPresent(material, "_BaseMap", baseColor);
            SetColorIfPresent(material, "_Color", Color.white);
            SetColorIfPresent(material, "_BaseColor", Color.white);

            SetTextureIfPresent(material, "_BumpMap", normal);
            SetFloatIfPresent(material, "_BumpScale", 1.0f);

            SetTextureIfPresent(material, "_MetallicGlossMap", metallicSmoothness);
            SetFloatIfPresent(material, "_Metallic", 1.0f);
            SetFloatIfPresent(material, "_Smoothness", 1.0f);
            SetFloatIfPresent(material, "_Glossiness", 1.0f);
            SetFloatIfPresent(material, "_GlossMapScale", 1.0f);
            SetFloatIfPresent(material, "_WorkflowMode", 1.0f);

            material.EnableKeyword("_NORMALMAP");
            material.EnableKeyword("_METALLICGLOSSMAP");
            material.EnableKeyword("_METALLICSPECGLOSSMAP");
        }

        private static void AssignMaterials(GameObject root, string folder)
        {
            var faces = DiscoverFaces(folder);
            if (faces.Length == 0)
            {
                return;
            }
            foreach (var renderer in root.GetComponentsInChildren<Renderer>(true))
            {
                var sharedMaterials = renderer.sharedMaterials;
                var changed = false;

                for (var i = 0; i < sharedMaterials.Length; i++)
                {
                    var face = GuessFace(sharedMaterials[i] != null ? sharedMaterials[i].name : string.Empty, i, faces);
                    var material = AssetDatabase.LoadAssetAtPath<Material>($"{folder}/Materials/{face}.mat");
                    if (material == null)
                    {
                        continue;
                    }

                    sharedMaterials[i] = material;
                    changed = true;
                }

                if (changed)
                {
                    renderer.sharedMaterials = sharedMaterials;
                }
            }
        }

        private static void BuildPrefab(string folder, string roomPath)
        {
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(roomPath);
            if (model == null)
            {
                return;
            }

            var instance = (GameObject)PrefabUtility.InstantiatePrefab(model);
            if (instance == null)
            {
                return;
            }

            AssignMaterials(instance, folder);
            PrefabUtility.SaveAsPrefabAsset(instance, $"{folder}/room_pbr.prefab");
            UnityEngine.Object.DestroyImmediate(instance);
        }

        private static Texture2D LoadTexture(string path)
        {
            return AssetDatabase.LoadAssetAtPath<Texture2D>(Normalize(path));
        }

        private static void SetTextureIfPresent(Material material, string property, Texture texture)
        {
            if (texture != null && material.HasProperty(property))
            {
                material.SetTexture(property, texture);
            }
        }

        private static void SetFloatIfPresent(Material material, string property, float value)
        {
            if (material.HasProperty(property))
            {
                material.SetFloat(property, value);
            }
        }

        private static void SetColorIfPresent(Material material, string property, Color value)
        {
            if (material.HasProperty(property))
            {
                material.SetColor(property, value);
            }
        }

        private static string[] DiscoverFaces(string folder)
        {
            var textureFolder = $"{folder}/textures";
            return AssetDatabase.FindAssets("t:Texture2D", new[] { textureFolder })
                .Select(AssetDatabase.GUIDToAssetPath)
                .Where(path => Normalize(Path.GetDirectoryName(path) ?? string.Empty) == textureFolder)
                .Select(Path.GetFileNameWithoutExtension)
                .Where(name => !string.IsNullOrWhiteSpace(name))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(FaceOrder)
                .ThenBy(name => name, StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        private static int FaceOrder(string face)
        {
            if (string.Equals(face, "floor", StringComparison.OrdinalIgnoreCase)) return 0;
            if (string.Equals(face, "ceiling", StringComparison.OrdinalIgnoreCase)) return 1;
            if (face.StartsWith("wall_", StringComparison.OrdinalIgnoreCase) &&
                int.TryParse(face.Substring(5), out var wallIndex)) return 100 + wallIndex;
            return 10000;
        }

        private static string GuessFace(string materialName, int materialIndex, string[] faces)
        {
            var clean = (materialName ?? string.Empty)
                .Replace(" (Instance)", string.Empty)
                .Trim();

            foreach (var face in faces)
            {
                if (string.Equals(clean, face, StringComparison.OrdinalIgnoreCase))
                {
                    return face;
                }
            }

            return materialIndex >= 0 && materialIndex < faces.Length ? faces[materialIndex] : faces[0];
        }

        private static bool IsRoomObj(string path)
        {
            var clean = Normalize(path);
            return IsInPackage(clean) && clean.EndsWith("/room.obj", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsInPackage(string path)
        {
            var clean = Normalize(path);
            if (clean.Contains($"/{PackageMarker}/") || clean.Contains($"{PackageMarker}/"))
            {
                return true;
            }

            // Also identify renamed copies by their aligned PBR directory
            // layout, so automatic and manual setup keep working after import.
            var root = Normalize(Path.GetDirectoryName(clean) ?? string.Empty);
            var pbrIndex = clean.IndexOf("/pbr_textures/", StringComparison.OrdinalIgnoreCase);
            var textureIndex = clean.IndexOf("/textures/", StringComparison.OrdinalIgnoreCase);
            if (pbrIndex >= 0)
            {
                root = clean.Substring(0, pbrIndex);
            }
            else if (textureIndex >= 0)
            {
                root = clean.Substring(0, textureIndex);
            }
            return AssetDatabase.IsValidFolder($"{root}/pbr_textures/normal") &&
                   AssetDatabase.IsValidFolder($"{root}/pbr_textures/unity_metallic_smoothness");
        }

        private static string Normalize(string path)
        {
            return (path ?? string.Empty).Replace("\\", "/");
        }
    }
}
#endif
