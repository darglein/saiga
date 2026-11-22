/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/WorldSettings.h"

#include <algorithm>

namespace Saiga
{
namespace Vulkan
{
namespace
{
template <typename S, typename F>
void forEachSetting(S& s, F f)
{
    f("shading", s.shading);
    f("terrain", s.terrain);
    f("trees", s.trees);
    f("grass", s.grass);
    f("water", s.water);
    f("sky", s.sky);
    f("depth_pre_pass", s.depthPrePass);
    f("debug_mode", s.debugMode);
    f("flat_shading", s.flatShading);
    f("fovy", s.fovy);
    f("albedo", s.albedo);
    f("wind", s.wind);
    f("terrain_lod_range", s.terrainLodRange);
    f("morph_start", s.morphStart);
    f("tree_lod1", s.treeLod1);
    f("tree_lod2", s.treeLod2);
    f("tree_max_distance", s.treeMaxDistance);
    f("grass_radius", s.grassRadius);
    f("grass_cell_size", s.grassCellSize);
    f("grass_density", s.grassDensity);
    f("alpha_cutoff", s.alphaCutoff);
}
}  // namespace

json11::Json WorldSettings::toJson() const
{
    json11::Json::object o;
    forEachSetting(*this,
                   [&](const char* name, const auto& v)
                   {
                       using T = std::decay_t<decltype(v)>;
                       if constexpr (std::is_same_v<T, WorldShading>)
                           o[name] = json11::Json("clay");
                       else
                           o[name] = json11::Json(v);
                   });
    return o;
}

std::vector<std::string> WorldSettings::fromJson(const json11::Json& json)
{
    std::vector<std::string> unknown;
    for (auto& kv : json.object_items())
    {
        bool found = false;
        forEachSetting(*this,
                       [&](const char* name, auto& v)
                       {
                           if (kv.first != name) return;
                           found   = true;
                           using T = std::decay_t<decltype(v)>;
                           if constexpr (std::is_same_v<T, WorldShading>)
                           {
                               // Clay is the only shading mode
                               if (kv.second.string_value() == "clay")
                                   v = WorldShading::Clay;
                               else
                                   found = false;
                           }
                           else if constexpr (std::is_same_v<T, bool>)
                           {
                               // A value of the wrong type is reported like an unknown key, not converted
                               if (kv.second.is_bool())
                                   v = kv.second.bool_value();
                               else
                                   found = false;
                           }
                           else if constexpr (std::is_same_v<T, int>)
                           {
                               // int_value() is undefined outside the int range: bound the number first, clamp()
                               // then limits it to the GUI range (debug_mode 1001 -> 1, not 0)
                               if (kv.second.is_number())
                                   v = int(std::max(-1e6, std::min(kv.second.number_value(), 1e6)));
                               else
                                   found = false;
                           }
                           else
                           {
                               // The float conversion of a double outside the float range is undefined
                               if (kv.second.is_number())
                                   v = float(std::max(-1e30, std::min(kv.second.number_value(), 1e30)));
                               else
                                   found = false;
                           }
                       });
        if (!found) unknown.push_back(kv.first);
    }
    clamp();
    return unknown;
}

void WorldSettings::clamp()
{
    // The ranges of the GUI sliders. Values from MCP are not limited otherwise; e.g. a tiny grass cell size would draw
    // billions of instances (device lost). NaN becomes the lower bound.
    auto c    = [](float& v, float lo, float hi) { v = v >= lo ? std::min(v, hi) : lo; };
    debugMode = std::min(std::max(debugMode, 0), 1);
    c(fovy, 20, 110);
    c(albedo, 0, 1);
    c(wind, 0, 4);
    c(terrainLodRange, 100, 2000);
    c(morphStart, 0.3f, 0.95f);
    c(treeLod1, 10, 500);
    c(treeLod2, 50, 3000);
    c(treeMaxDistance, 500, 50000);
    c(grassRadius, 0, 200);
    c(grassCellSize, 0.1f, 2.0f);
    c(grassDensity, 0, 1);
    c(alphaCutoff, 0.05f, 0.95f);
}

PrototypeSettings WorldSettings::prototype() const
{
    PrototypeSettings p;
    p.terrain         = terrain;
    p.grass           = grass;
    p.sky             = sky;
    p.water           = water;
    p.depthPrePass    = depthPrePass;
    p.albedo          = albedo;
    p.wind            = wind;
    p.terrainLodRange = terrainLodRange;
    p.morphStart      = morphStart;
    p.grassRadius     = grassRadius;
    p.grassCellSize   = grassCellSize;
    p.grassDensity    = grassDensity;
    p.alphaCutoff     = alphaCutoff;
    p.debugMode       = debugMode;
    p.flatShading     = flatShading;
    return p;
}

}  // namespace Vulkan
}  // namespace Saiga
