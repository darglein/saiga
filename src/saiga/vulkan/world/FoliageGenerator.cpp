/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/FoliageGenerator.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>

namespace Saiga
{
namespace Vulkan
{
namespace
{
constexpr float pi = 3.14159265358979f;

vec3 Y()
{
    return vec3(0, 1, 0);
}

struct MeshBuilder
{
    FoliageMesh& mesh;
    std::mt19937 rng;

    MeshBuilder(FoliageMesh& mesh, uint32_t seed) : mesh(mesh), rng(seed) {}

    float rnd(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

    void vertex(const vec3& p, const vec3& n, float ao, float wind, vec2 uv = vec2(0, 0), float layer = -1)
    {
        TreeVertex v;
        v.posAo      = vec4(p(0), p(1), p(2), ao);
        v.normalWind = vec4(n(0), n(1), n(2), wind);
        v.uvLayer    = vec4(uv(0), uv(1), layer, 0);
        mesh.vertices.push_back(v);
    }

    // Flat shaded triangle, oriented away from 'inside'. ao and wind are per vertex.
    void tri(vec3 a, vec3 b, vec3 c, const vec3& inside, vec3 ao, vec3 wind)
    {
        vec3 n = (b - a).cross(c - a);
        if (n.norm() < 1e-9f) return;
        n.normalize();
        if (n.dot((a + b + c) / 3.0f - inside) < 0)
        {
            std::swap(b, c);
            std::swap(ao(1), ao(2));
            std::swap(wind(1), wind(2));
            n = -n;
        }
        uint32_t base = uint32_t(mesh.vertices.size());
        vertex(a, n, ao(0), wind(0));
        vertex(b, n, ao(1), wind(1));
        vertex(c, n, ao(2), wind(2));
        mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2});
    }

    static vec3 ring(float angle) { return vec3(std::cos(angle), 0, std::sin(angle)); }

    // A (slightly irregular) prism along the axis b -> t
    void prism(const vec3& b, const vec3& t, float rb, float rt, int sides, float aoB, float aoT, float windB,
               float windT)
    {
        vec3 center = (b + t) * 0.5f;
        float phase = rnd(0, 2 * pi);
        for (int k = 0; k < sides; ++k)
        {
            float a0 = phase + 2 * pi * k / sides, a1 = phase + 2 * pi * (k + 1) / sides;
            vec3 p0 = b + ring(a0) * rb, p1 = b + ring(a1) * rb;
            vec3 q0 = t + ring(a0) * rt, q1 = t + ring(a1) * rt;
            tri(p0, p1, q1, center, vec3(aoB, aoB, aoT), vec3(windB, windB, windT));
            tri(p0, q1, q0, center, vec3(aoB, aoT, aoT), vec3(windB, windT, windT));
        }
    }

    // Cone with a bottom cap
    void cone(const vec3& base, float h, float r, int sides, float aoBase, float aoTip, float windBase, float windTip)
    {
        vec3 apex   = base + Y() * h + vec3(rnd(-0.01f, 0.01f), 0, rnd(-0.01f, 0.01f));
        vec3 in     = base + Y() * (h * 0.3f);
        float phase = rnd(0, 2 * pi);
        std::vector<vec3> pts;
        for (int k = 0; k < sides; ++k)
        {
            float a = phase + 2 * pi * k / sides + rnd(-0.15f, 0.15f);
            pts.push_back(base + ring(a) * (r * rnd(0.88f, 1.08f)) + Y() * rnd(-0.015f, 0.015f));
        }
        for (int k = 0; k < sides; ++k)
        {
            const vec3& p0 = pts[k];
            const vec3& p1 = pts[(k + 1) % sides];
            tri(p0, p1, apex, in, vec3(aoBase, aoBase, aoTip), vec3(windBase, windBase, windTip));
            // Cap (seen from below)
            tri(base, p1, p0, base + Y() * h, vec3(aoBase * 0.8f, aoBase * 0.8f, aoBase * 0.8f),
                vec3(windBase, windBase, windBase));
        }
    }

    // Jittered icosphere (subdiv 0: 20 triangles, 1: 80 triangles). radii: per axis. The bottom is darker.
    void blob(const vec3& c, const vec3& radii, int subdiv, float jitter, float windScale)
    {
        const float t       = (1.0f + std::sqrt(5.0f)) / 2.0f;
        std::vector<vec3> v = {vec3(-1, t, 0), vec3(1, t, 0), vec3(-1, -t, 0), vec3(1, -t, 0),
                               vec3(0, -1, t), vec3(0, 1, t), vec3(0, -1, -t), vec3(0, 1, -t),
                               vec3(t, 0, -1), vec3(t, 0, 1), vec3(-t, 0, -1), vec3(-t, 0, 1)};
        for (auto& p : v) p.normalize();
        std::vector<std::array<int, 3>> f = {{0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
                                             {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                             {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
                                             {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
        for (int s = 0; s < subdiv; ++s)
        {
            std::map<std::pair<int, int>, int> mid;
            auto midpoint = [&](int a, int b)
            {
                auto key = std::make_pair(std::min(a, b), std::max(a, b));
                auto it  = mid.find(key);
                if (it != mid.end()) return it->second;
                v.push_back((v[a] + v[b]).normalized());
                mid[key] = int(v.size()) - 1;
                return int(v.size()) - 1;
            };
            std::vector<std::array<int, 3>> nf;
            for (auto& tr : f)
            {
                int ab = midpoint(tr[0], tr[1]), bc = midpoint(tr[1], tr[2]), ca = midpoint(tr[2], tr[0]);
                nf.push_back({tr[0], ab, ca});
                nf.push_back({tr[1], bc, ab});
                nf.push_back({tr[2], ca, bc});
                nf.push_back({ab, bc, ca});
            }
            f = nf;
        }
        // Shared jitter per vertex keeps the blob closed
        std::vector<vec3> p(v.size());
        for (size_t i = 0; i < v.size(); ++i)
        {
            float j = 1.0f + rnd(-jitter, jitter);
            p[i]    = c + vec3(v[i](0) * radii(0), v[i](1) * radii(1), v[i](2) * radii(2)) * j;
        }
        auto ao = [&](const vec3& q)
        {
            float h = (q(1) - c(1)) / radii(1);
            return 0.55f + 0.45f * std::min(std::max(h * 0.5f + 0.5f, 0.0f), 1.0f);
        };
        auto wind = [&](const vec3& q) { return windScale * q(1); };
        for (auto& tr : f)
        {
            const vec3 &a = p[tr[0]], &b = p[tr[1]], &d = p[tr[2]];
            tri(a, b, d, c, vec3(ao(a), ao(b), ao(d)), vec3(wind(a), wind(b), wind(d)));
        }
    }

    // A two sided alpha tested quad. v = 0 is at the top (center + up * h / 2).
    void card(const vec3& center, const vec3& right, const vec3& up, float w, float h, int layer, const vec3& normal,
              float ao, float windBottom, float windTop)
    {
        uint32_t base = uint32_t(mesh.vertices.size());
        vec3 r = right * (w * 0.5f), u = up * (h * 0.5f);
        vertex(center - r + u, normal, ao, windTop, vec2(0, 0), float(layer));
        vertex(center + r + u, normal, ao, windTop, vec2(1, 0), float(layer));
        vertex(center + r - u, normal, ao * 0.85f, windBottom, vec2(1, 1), float(layer));
        vertex(center - r - u, normal, ao * 0.85f, windBottom, vec2(0, 1), float(layer));
        mesh.indices.insert(mesh.indices.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
    }

    // A card facing outwards from a blob center, with a random rotation in its plane
    void blobCard(const vec3& blobCenter, float r, float size, int layer, float windScale)
    {
        vec3 dir(rnd(-1, 1), rnd(-0.6f, 1.0f), rnd(-1, 1));
        if (dir.norm() < 1e-3f) dir = Y();
        dir.normalize();
        vec3 right = std::abs(dir(1)) < 0.95f ? Y().cross(dir).normalized() : vec3(1, 0, 0);
        vec3 up    = dir.cross(right).normalized();
        float a    = rnd(0, 2 * pi);
        vec3 r2    = right * std::cos(a) + up * std::sin(a);
        vec3 u2    = up * std::cos(a) - right * std::sin(a);
        // Tilt the card a bit out of the tangent plane so it is visible from more directions
        u2       = (u2 + dir * rnd(-0.5f, 0.5f)).normalized();
        vec3 c   = blobCenter + dir * (r * rnd(0.75f, 1.05f));
        vec3 n   = (dir + Y() * 0.3f).normalized();
        float ao = 0.75f + 0.25f * std::max(dir(1), 0.0f);
        card(c, r2, u2, size, size, layer, n, ao, windScale * c(1) * 0.8f, windScale * c(1) * 1.2f);
    }
};

TreeSpecies spruce(uint32_t seed)
{
    TreeSpecies s;
    s.name        = "spruce";
    s.heightRange = vec2(10, 22);
    for (int lod = 0; lod < 3; ++lod)
    {
        MeshBuilder b(s.lods[lod], seed);
        b.prism(vec3(0, -0.05f, 0), vec3(0, 0.3f, 0), 0.022f, 0.015f, lod == 2 ? 4 : (lod == 1 ? 5 : 6), 0.5f, 0.6f, 0,
                0.05f);
        if (lod == 2)
        {
            b.cone(vec3(0, 0.14f, 0), 0.88f, 0.27f, 5, 0.6f, 1.0f, 0.05f, 0.4f);
            continue;
        }
        const int tiers = 4;
        for (int i = 0; i < tiers; ++i)
        {
            float base = 0.16f + i * 0.185f;
            float h    = 0.38f - i * 0.05f;
            float r    = 0.27f - i * 0.055f;
            float ao0  = 0.55f + 0.1f * i;
            b.cone(vec3(0, base, 0), h, r, lod == 0 ? 8 : 6, ao0, 1.0f, 0.1f + 0.2f * i, 0.4f);

            // Needle cards along the cone surface, hanging over its edge
            int cards  = lod == 0 ? 9 : 4;
            float size = lod == 0 ? 1.0f : 1.35f;
            for (int k = 0; k < cards; ++k)
            {
                float a     = 2 * pi * k / cards + i * 0.6f + b.rnd(-0.2f, 0.2f);
                vec3 out    = MeshBuilder::ring(a);
                vec3 right  = vec3(-std::sin(a), 0, std::cos(a));
                vec3 slope  = (Y() * h - out * r).normalized();
                float slant = std::sqrt(r * r + h * h);
                vec3 center = vec3(0, base, 0) + out * (r * 0.62f) + Y() * (h * 0.36f) + out * 0.03f;
                float w     = 2 * pi * r / cards * 1.5f * size;
                vec3 n      = (out + Y() * 0.6f).normalized();
                b.card(center, right, slope, std::min(w, 0.32f), slant * 0.75f * size, FOLIAGE_NEEDLES, n,
                       0.7f + 0.08f * i, 0.15f + 0.2f * i, 0.35f + 0.2f * i);
            }
        }
    }
    return s;
}

TreeSpecies broadleaf(uint32_t seed)
{
    TreeSpecies s;
    s.name        = "broadleaf";
    s.heightRange = vec2(8, 14);
    struct Blob
    {
        vec3 c;
        float r;
    };
    std::vector<Blob> blobs = {{vec3(0, 0.7f, 0), 0.25f},
                               {vec3(0.18f, 0.56f, 0.08f), 0.18f},
                               {vec3(-0.16f, 0.6f, -0.1f), 0.19f},
                               {vec3(0.02f, 0.55f, -0.19f), 0.17f}};
    for (int lod = 0; lod < 3; ++lod)
    {
        MeshBuilder b(s.lods[lod], seed + 11);
        b.prism(vec3(0, -0.05f, 0), vec3(0, 0.45f, 0), 0.03f, 0.02f, lod == 2 ? 4 : 6, 0.5f, 0.6f, 0, 0.05f);
        if (lod == 2)
        {
            b.blob(vec3(0, 0.65f, 0), vec3(0.32f, 0.27f, 0.32f), 0, 0.12f, 0.4f);
            continue;
        }
        for (size_t i = 1; i < blobs.size(); ++i)
        {
            // Branches to the side blobs
            if (lod == 0)
            {
                b.prism(vec3(0, 0.36f, 0), blobs[i].c * 0.8f + vec3(0, 0.1f, 0), 0.012f, 0.007f, 4, 0.55f, 0.6f, 0.03f,
                        0.2f);
            }
        }
        for (auto& bl : blobs)
        {
            // A solid core inside of the leaf cards
            float core = 0.72f;
            b.blob(bl.c, vec3(bl.r, bl.r * 0.85f, bl.r) * core, lod == 0 ? 1 : 0, 0.15f, 0.4f);
            int cards = lod == 0 ? 20 : 6;
            for (int k = 0; k < cards; ++k)
            {
                b.blobCard(bl.c, bl.r, bl.r * (lod == 0 ? 1.1f : 1.5f), FOLIAGE_BROADLEAF, 0.5f);
            }
        }
    }
    return s;
}

TreeSpecies birch(uint32_t seed)
{
    TreeSpecies s;
    s.name        = "birch";
    s.heightRange = vec2(10, 17);
    struct Blob
    {
        vec3 c;
        float r;
    };
    std::vector<Blob> blobs = {{vec3(0, 0.56f, 0), 0.14f},
                               {vec3(0.02f, 0.72f, 0.02f), 0.15f},
                               {vec3(0.03f, 0.88f, -0.01f), 0.11f},
                               {vec3(-0.07f, 0.64f, 0.07f), 0.12f}};
    for (int lod = 0; lod < 3; ++lod)
    {
        MeshBuilder b(s.lods[lod], seed + 23);
        b.prism(vec3(0, -0.05f, 0), vec3(0, 0.62f, 0), 0.018f, 0.011f, lod == 2 ? 4 : 6, 0.7f, 0.75f, 0, 0.1f);
        if (lod == 2)
        {
            b.blob(vec3(0, 0.71f, 0), vec3(0.16f, 0.25f, 0.16f), 0, 0.12f, 0.5f);
            continue;
        }
        for (auto& bl : blobs)
        {
            float core = 0.72f;
            b.blob(bl.c, vec3(bl.r, bl.r * 1.15f, bl.r) * core, lod == 0 ? 1 : 0, 0.15f, 0.5f);
            int cards = lod == 0 ? 16 : 5;
            for (int k = 0; k < cards; ++k)
            {
                b.blobCard(bl.c, bl.r, bl.r * (lod == 0 ? 1.15f : 1.5f), FOLIAGE_BROADLEAF, 0.6f);
            }
        }
    }
    return s;
}

// Coverage canvas, the shapes are drawn with an antialiased edge (max blending)
struct Canvas
{
    int size;
    std::vector<float> a;

    explicit Canvas(int size) : size(size), a(size_t(size) * size, 0.0f) {}

    // f(p) returns the signed distance in uv units at p (uv in [0, 1]^2)
    template <typename F>
    void draw(vec2 bmin, vec2 bmax, F f)
    {
        int x0 = std::max(0, int(bmin(0) * size) - 1), x1 = std::min(size - 1, int(bmax(0) * size) + 1);
        int y0 = std::max(0, int(bmin(1) * size) - 1), y1 = std::min(size - 1, int(bmax(1) * size) + 1);
        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x)
            {
                vec2 p((x + 0.5f) / size, (y + 0.5f) / size);
                float d     = f(p);
                float alpha = std::min(std::max(0.5f - d * size, 0.0f), 1.0f);
                float& dst  = a[size_t(y) * size + x];
                dst         = std::max(dst, alpha);
            }
        }
    }

    void segment(vec2 p0, vec2 p1, float w0, float w1)
    {
        float m   = std::max(w0, w1);
        vec2 bmin = vec2(std::min(p0(0), p1(0)) - m, std::min(p0(1), p1(1)) - m);
        vec2 bmax = vec2(std::max(p0(0), p1(0)) + m, std::max(p0(1), p1(1)) + m);
        vec2 d    = p1 - p0;
        float dd  = std::max(d.dot(d), 1e-12f);
        draw(bmin, bmax,
             [&](const vec2& p)
             {
                 float t = std::min(std::max((p - p0).dot(d) / dd, 0.0f), 1.0f);
                 return (p - (p0 + d * t)).norm() - (w0 + (w1 - w0) * t) * 0.5f;
             });
    }

    // Pointed leaf along dir with the half length len and the half width wid
    void leaf(vec2 c, vec2 dir, float len, float wid)
    {
        vec2 perp(-dir(1), dir(0));
        draw(c - vec2(len, len), c + vec2(len, len),
             [&](const vec2& p)
             {
                 vec2 q  = p - c;
                 float u = q.dot(dir) / len, v = q.dot(perp);
                 // Narrower at the tips
                 float w = wid * std::max(1.0f - u * u, 0.05f);
                 float r = std::sqrt(u * u + (v / w) * (v / w));
                 return (r - 1.0f) * std::min(len, w);
             });
    }

    TemplatedImage<unsigned char> image() const
    {
        TemplatedImage<unsigned char> img(size, size);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) img(y, x) = (unsigned char)std::lround(a[size_t(y) * size + x] * 255.0f);
        return img;
    }
};

}  // namespace

std::vector<TreeSpecies> FoliageGenerator::treeSpecies(uint32_t seed)
{
    std::vector<TreeSpecies> species = {spruce(seed), broadleaf(seed), birch(seed)};
    for (auto& s : species)
    {
        float ymin = 1e9f, ymax = -1e9f;
        for (auto& m : s.lods)
            for (auto& v : m.vertices)
            {
                ymin = std::min(ymin, v.posAo(1));
                ymax = std::max(ymax, v.posAo(1));
            }
        float cy = (ymin + ymax) * 0.5f, r = 0;
        for (auto& m : s.lods)
            for (auto& v : m.vertices) r = std::max(r, (make_vec3(v.posAo) - vec3(0, cy, 0)).norm());
        s.bounds = vec2(cy, r);
    }
    return species;
}

FoliageMesh FoliageGenerator::grassClump()
{
    FoliageMesh m;
    MeshBuilder b(m, 1);
    for (int k = 0; k < 3; ++k)
    {
        float a = k * pi / 3.0f;
        vec3 right(std::cos(a), 0, std::sin(a));
        uint32_t base = uint32_t(m.vertices.size());
        // Bottom at y = 0, top at y = 1. The wind weight is used as "height in the clump" by the shader.
        b.vertex(-right * 0.5f + Y(), Y(), 1.0f, 1.0f, vec2(0, 0), FOLIAGE_GRASS);
        b.vertex(right * 0.5f + Y(), Y(), 1.0f, 1.0f, vec2(1, 0), FOLIAGE_GRASS);
        b.vertex(right * 0.5f, Y(), 0.45f, 0.0f, vec2(1, 1), FOLIAGE_GRASS);
        b.vertex(-right * 0.5f, Y(), 0.45f, 0.0f, vec2(0, 1), FOLIAGE_GRASS);
        m.indices.insert(m.indices.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
    }
    return m;
}

std::vector<TemplatedImage<unsigned char>> FoliageGenerator::foliageTextures(int size, uint32_t seed)
{
    std::mt19937 rng(seed);
    auto rnd = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };

    std::vector<TemplatedImage<unsigned char>> layers;

    // Broadleaf: a cluster of leaves pointing outwards, with a few twigs
    {
        Canvas c(size);
        vec2 center(0.5f, 0.5f);
        for (int k = 0; k < 6; ++k)
        {
            float a = rnd(0, 2 * pi);
            c.segment(center, center + vec2(std::cos(a), std::sin(a)) * rnd(0.2f, 0.32f), 0.012f, 0.005f);
        }
        for (int k = 0; k < 34; ++k)
        {
            float len = rnd(0.085f, 0.12f), wid = rnd(0.035f, 0.05f);
            float a  = rnd(0, 2 * pi);
            float r  = std::sqrt(rnd(0, 1)) * (0.46f - len);
            vec2 p   = center + vec2(std::cos(a), std::sin(a)) * r;
            float da = a + rnd(-0.6f, 0.6f);
            c.leaf(p, vec2(std::cos(da), std::sin(da)), len, wid);
        }
        layers.push_back(c.image());
    }

    // Needles: a twig from the top to the bottom with side twigs, needles pointing down and outwards
    {
        Canvas c(size);
        struct Twig
        {
            vec2 p0, p1;
        };
        std::vector<Twig> twigs = {{vec2(0.5f, 0.03f), vec2(0.5f, 0.97f)}};
        for (float t : {0.22f, 0.45f, 0.68f})
        {
            for (float side : {-1.0f, 1.0f})
            {
                vec2 s = vec2(0.5f, 0.03f + 0.94f * (t + rnd(-0.04f, 0.04f)));
                vec2 e = s + vec2(side * rnd(0.22f, 0.3f) * (1.1f - t * 0.5f), rnd(0.12f, 0.2f));
                twigs.push_back({s, e});
            }
        }
        for (size_t k = 0; k < twigs.size(); ++k)
        {
            auto& tw = twigs[k];
            c.segment(tw.p0, tw.p1, k == 0 ? 0.016f : 0.01f, 0.006f);
            vec2 d    = tw.p1 - tw.p0;
            float len = d.norm();
            vec2 dir  = d / len;
            vec2 perp(-dir(1), dir(0));
            for (float s = 0.02f; s < len; s += 0.022f)
            {
                vec2 p = tw.p0 + dir * s;
                for (float side : {-1.0f, 1.0f})
                {
                    // 55 degrees from the twig, towards its end
                    vec2 nd    = (dir * 0.57f + perp * side * 0.82f).normalized();
                    float nlen = rnd(0.06f, 0.09f) * (1.0f - 0.4f * s / len);
                    c.segment(p, p + nd * nlen, 0.008f, 0.004f);
                }
            }
        }
        layers.push_back(c.image());
    }

    // Grass: tapered, slightly bent blades from the bottom edge
    {
        Canvas c(size);
        int blades = 11;
        for (int k = 0; k < blades; ++k)
        {
            float bx  = 0.1f + 0.8f * (k + rnd(-0.3f, 0.3f)) / (blades - 1);
            vec2 base = vec2(bx, 0.995f);
            vec2 tip  = vec2(bx + rnd(-0.15f, 0.15f), rnd(0.04f, 0.35f));
            vec2 ctrl = vec2(bx + rnd(-0.05f, 0.05f), (base(1) + tip(1)) * 0.5f);
            float w0  = rnd(0.035f, 0.05f);
            int segs  = 8;
            vec2 prev = base;
            for (int s = 1; s <= segs; ++s)
            {
                float t = float(s) / segs;
                // Quadratic bezier
                vec2 p = base * ((1 - t) * (1 - t)) + ctrl * (2 * (1 - t) * t) + tip * (t * t);
                c.segment(prev, p, w0 * (1.0f - (t - 1.0f / segs)), w0 * (1.0f - t) + 0.002f);
                prev = p;
            }
        }
        layers.push_back(c.image());
    }
    return layers;
}

}  // namespace Vulkan
}  // namespace Saiga
