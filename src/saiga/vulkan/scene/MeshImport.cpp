/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/MeshImport.h"

#include "saiga/saiga_modules.h"

#if defined(SAIGA_USE_ASSIMP)
#    include <assimp/GltfMaterial.h>
#    include <assimp/Importer.hpp>
#    include <assimp/material.h>
#    include <assimp/scene.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace Saiga
{
namespace Vulkan
{
#if !defined(SAIGA_USE_ASSIMP)
bool importMeshLods(const std::string& file, MeshLods&, MaterialData*, std::string& error, const MeshImportOptions&)
{
    error = file + ": importMeshLods needs assimp (SAIGA_WITH_ASSIMP)";
    return false;
}
#else
namespace
{
// At most this many vertices per LOD (after the face duplication)
constexpr size_t maxVertices = size_t(1) << 24;

// The LOD index of a node name ending in LOD<k>, else -1
int lodSuffix(const std::string& name)
{
    size_t digits = name.size();
    while (digits > 0 && std::isdigit(static_cast<unsigned char>(name[digits - 1]))) digits--;
    if (digits == name.size() || digits < 3 || name.compare(digits - 3, 3, "LOD") != 0) return -1;
    if (name.size() - digits > 3) return 1000;
    return std::stoi(name.substr(digits));
}

struct MeshItem
{
    unsigned mesh;
    aiMatrix4x4 transform;
    int lod;
    std::string node;
};

// The properties of a glTF material that must agree within the merged mesh
struct MaterialProps
{
    std::string name;
    bool alphaMask = false, twoSided = false;
    aiColor3D emissive = aiColor3D(0, 0, 0);
    float roughness = 1, metalness = 1;
    aiColor4D baseColor = aiColor4D(1, 1, 1, 1);

    bool sameAs(const MaterialProps& o) const
    {
        return alphaMask == o.alphaMask && twoSided == o.twoSided && emissive == o.emissive &&
               roughness == o.roughness && metalness == o.metalness;
    }
};

MaterialProps readMaterial(const aiMaterial* m)
{
    MaterialProps p;
    aiString name;
    if (m->Get(AI_MATKEY_NAME, name) == AI_SUCCESS) p.name = name.C_Str();
    aiString mode;
    if (m->Get(AI_MATKEY_GLTF_ALPHAMODE, mode) == AI_SUCCESS) p.alphaMask = std::string(mode.C_Str()) == "MASK";
    int twoSided = 0;
    if (m->Get(AI_MATKEY_TWOSIDED, twoSided) == AI_SUCCESS) p.twoSided = twoSided != 0;
    m->Get(AI_MATKEY_COLOR_EMISSIVE, p.emissive);
    m->Get(AI_MATKEY_ROUGHNESS_FACTOR, p.roughness);
    m->Get(AI_MATKEY_METALLIC_FACTOR, p.metalness);
    if (m->Get(AI_MATKEY_BASE_COLOR, p.baseColor) != AI_SUCCESS) m->Get(AI_MATKEY_COLOR_DIFFUSE, p.baseColor);
    return p;
}

void collect(const aiNode* node, const aiMatrix4x4& parent, int lod, std::vector<MeshItem>& items,
             std::vector<int>& lodNodes, std::string& error)
{
    const aiMatrix4x4 transform = parent * node->mTransformation;
    const std::string name      = node->mName.C_Str();
    const int suffix            = lodSuffix(name);
    if (suffix >= 0)
    {
        if (lod >= 0 && error.empty()) error = "the LOD node " + name + " is inside another LOD node";
        lod = suffix;
        lodNodes.push_back(suffix);
    }
    for (unsigned i = 0; i < node->mNumMeshes; ++i) items.push_back({node->mMeshes[i], transform, lod, name});
    for (unsigned i = 0; i < node->mNumChildren; ++i)
        collect(node->mChildren[i], transform, lod, items, lodNodes, error);
}

float linearToSrgb(float c)
{
    c = std::clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

uint8_t unorm8(float v)
{
    return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
}
}  // namespace

bool importMeshLods(const std::string& file, MeshLods& out, MaterialData* material, std::string& error,
                    const MeshImportOptions& options)
{
    error.clear();
    auto fail = [&](const std::string& message)
    {
        error = file + ": " + message;
        return false;
    };

    // No post-processing: no triangulation, no joined vertices (the faces are duplicated anyway)
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(file, 0);
    if (!scene || !scene->mRootNode) return fail(std::string("could not be read: ") + importer.GetErrorString());

    std::vector<MeshItem> items;
    std::vector<int> lodNodes;
    std::string structure;
    collect(scene->mRootNode, aiMatrix4x4(), -1, items, lodNodes, structure);
    if (!structure.empty()) return fail(structure);
    if (items.empty()) return fail("no mesh");

    int lodCount = 1;
    if (!lodNodes.empty())
    {
        lodCount = *std::max_element(lodNodes.begin(), lodNodes.end()) + 1;
        if (lodCount > 4) return fail("at most 4 LODs (LOD0 .. LOD3)");
        for (int l = 0; l < lodCount; ++l)
            if (std::find(lodNodes.begin(), lodNodes.end(), l) == lodNodes.end())
                return fail("the LOD suffixes are not contiguous from LOD0 (LOD" + std::to_string(l) + " is missing)");
        for (const MeshItem& it : items)
            if (it.lod < 0) return fail("the mesh node " + it.node + " is outside the LOD nodes");
    }
    else
    {
        for (MeshItem& it : items) it.lod = 0;
    }

    // The material of the merged mesh
    const MaterialProps first = readMaterial(scene->mMaterials[scene->mMeshes[items.front().mesh]->mMaterialIndex]);
    for (const MeshItem& it : items)
    {
        const MaterialProps p = readMaterial(scene->mMaterials[scene->mMeshes[it.mesh]->mMaterialIndex]);
        if (!p.sameAs(first))
            return fail("the merged primitives have different materials: " + first.name + " and " + p.name +
                        " (alphaMode, doubleSided, emissive, roughness or metalness differ)");
    }

    MeshLods result;
    result.lods.resize(lodCount);
    for (int l = 0; l < lodCount; ++l) result.lods[l].endDistance = float(l + 1) * 1000.0f;

    for (const MeshItem& it : items)
    {
        const aiMesh* mesh = scene->mMeshes[it.mesh];
        if (mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE)
            return fail("the mesh " + std::string(mesh->mName.C_Str()) + " of node " + it.node +
                        " has points or lines (only triangles are supported)");
        const MaterialProps props = readMaterial(scene->mMaterials[mesh->mMaterialIndex]);
        MeshLod& lod              = result.lods[it.lod];

        const aiMatrix4x4& T = it.transform;
        const bool flip      = T.Determinant() < 0;
        // A COLOR_0 without alpha (VEC3): assimp delivers alpha 0 for all of them
        bool vec3Color = mesh->HasVertexColors(0);
        for (unsigned v = 0; vec3Color && v < mesh->mNumVertices; ++v) vec3Color = mesh->mColors[0][v].a == 0;

        auto colorWord = [&](unsigned v, uint32_t& word) -> bool
        {
            aiColor4D c(1, 1, 1, 1);
            if (mesh->HasVertexColors(0))
            {
                c = mesh->mColors[0][v];
                if (vec3Color) c.a = 1;
            }
            // Linear * linear, then the sRGB curve
            const float r = c.r * props.baseColor.r, g = c.g * props.baseColor.g, b = c.b * props.baseColor.b;
            const float a = props.alphaMask ? c.a * props.baseColor.a : 1.0f;
            ucvec4 rgba(unorm8(linearToSrgb(r)), unorm8(linearToSrgb(g)), unorm8(linearToSrgb(b)), unorm8(a));
            if (!options.palette)
            {
                word = packColor(rgba);
                return true;
            }
            const auto& pal = *options.palette;
            for (size_t i = 0; i < pal.size(); ++i)
            {
                if (pal[i](0) == rgba(0) && pal[i](1) == rgba(1) && pal[i](2) == rgba(2))
                {
                    word = uint32_t(i);
                    return true;
                }
            }
            std::ostringstream str;
            str << "the color (" << int(rgba(0)) << ", " << int(rgba(1)) << ", " << int(rgba(2))
                << ") (sRGB8) is not in the palette";
            fail(str.str());
            return false;
        };

        for (unsigned f = 0; f < mesh->mNumFaces; ++f)
        {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) return fail("a face of " + it.node + " is not a triangle");
            unsigned idx[3] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
            if (flip) std::swap(idx[1], idx[2]);
            vec3 p[3];
            for (int k = 0; k < 3; ++k)
            {
                const aiVector3D q = T * mesh->mVertices[idx[k]];
                p[k]               = vec3(q.x, q.y, q.z);
            }
            vec3 n          = (p[1] - p[0]).cross(p[2] - p[0]);
            const float len = n.norm();
            // Zero-area faces have no normal
            if (!(len > 0) || !std::isfinite(len)) continue;
            n /= len;
            const uint32_t normal = packNormalOct(n);
            for (int k = 0; k < 3; ++k)
            {
                MeshVertex v;
                v.position = p[k];
                v.normal   = normal;
                if (!colorWord(idx[k], v.color)) return false;
                if (mesh->HasTextureCoords(0))
                    v.uv = vec2(mesh->mTextureCoords[0][idx[k]].x, mesh->mTextureCoords[0][idx[k]].y);
                lod.indices.push_back(uint32_t(lod.vertices.size()));
                lod.vertices.push_back(v);
            }
            if (lod.vertices.size() > maxVertices) return fail("too many vertices (more than 2^24 in a LOD)");
        }
    }
    for (int l = 0; l < lodCount; ++l)
        if (result.lods[l].vertices.empty()) return fail("LOD" + std::to_string(l) + " has no (non-degenerate) face");

    // The bounding sphere of all LODs (around the AABB's centre)
    vec3 lo = result.lods[0].vertices[0].position, hi = lo;
    for (const MeshLod& l : result.lods)
        for (const MeshVertex& v : l.vertices)
        {
            lo = lo.cwiseMin(v.position);
            hi = hi.cwiseMax(v.position);
        }
    result.sphereCenter = (lo + hi) * 0.5f;
    float r2            = 0;
    for (const MeshLod& l : result.lods)
        for (const MeshVertex& v : l.vertices) r2 = std::max(r2, (v.position - result.sphereCenter).squaredNorm());
    result.sphereRadius = std::sqrt(r2);

    if (material)
    {
        MaterialData m;
        m.model      = ShadingModel::FlatColor;
        m.usePalette = options.palette != nullptr;
        m.baseColor  = vec4(1, 1, 1, 1);
        m.roughness  = first.roughness;
        m.metalness  = first.metalness;
        m.emissive   = vec3(first.emissive.r, first.emissive.g, first.emissive.b);
        m.alphaMask  = first.alphaMask;
        m.twoSided   = first.twoSided;
        *material    = m;
    }
    out = std::move(result);
    return true;
}
#endif

}  // namespace Vulkan
}  // namespace Saiga
