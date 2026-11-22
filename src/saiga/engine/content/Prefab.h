/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Prefab (docs/engine_design.md §14, §15.3; S23a): an authored place, a versioned JSON file
// <WorldSetup::dataDirectory>/prefabs/<name>.json (version 1) or <name>.v<N>.json (version N), read once (never
// reloaded, not a data file of §14). Everything is integer after parsing: rectangles in integer metres (half open
// [x0, z0, x1, z1)), positions in 1/256 m (x, z) and height units (y, 1/1024 m), yaws in 65536 steps per turn.
// Component values stay JSON in the visit form; they are checked when the plans are built (phase B of the generation
// pipeline) and applied after configure() / onSpawn() like the world API's spawn.
//
// Format 1:
//   "format": 1                                  required
//   "footprint": [x0, z0, x1, z1]                required, integer metres; everything below lies inside it
//   "terrain_patch": {"origin": [x, z], "size": [sx, sz], "heights": [sx * sz ints], "weights": [sx * sz ints]}
//                                                samples 1 m apart, row-major z then x; heights in height units
//                                                relative to the site's baseHeight; weights 0..256 (256 = patch)
//   "exclusion": [[x0, z0, x1, z1], ...]         no tier-1 candidates and no grass inside
//   "entities": [{"class": name, "position": [x, y, z], "yaw_deg": int, "components": {name: {fields}},
//                 "space_link": {"to": "link" | "parent", "link": n, "arrival": [x, y, z], "arrival_yaw_deg": int}}]
//   "static_geometry": [{"mesh": name, "position": [x, y, z], "yaw_deg": int}]
//   "spaces": [{"template": name}]               the space instances a site creates (space-0 sites only)
//   "pieces"                                     an error until S30
// Positions: x and z exact multiples of 1/256 m, y of 1/1024 m (an inexact value such as 0.1 is an error).

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/export.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
struct PrefabRect
{
    int32_t x0 = 0, z0 = 0, x1 = 0, z1 = 0;
};

struct PrefabPatch
{
    int32_t originX = 0, originZ = 0, sizeX = 0, sizeZ = 0;
    std::vector<int32_t> heights, weights;  // sizeX * sizeZ, row-major z then x
};

struct PrefabSpaceLink
{
    bool toParent = false;  // "parent": back to the space that created this instance; else the site's n-th space
    uint8_t link  = 0;
    // "link": in the target space's coordinates; "parent": relative to the creating site's origin, rotation and base
    // height. x, z in 1/256 m, y in height units
    int64_t x = 0, z = 0;
    int32_t y           = 0;
    uint16_t arrivalYaw = 0;
};

struct PrefabEntity
{
    std::string className;
    int64_t x = 0, z = 0;  // 1/256 m, prefab local
    int32_t y    = 0;      // height units relative to the site's baseHeight
    uint16_t yaw = 0;
    json11::Json components;  // an object {component: {fields}} in the visit form (unmodified)
    std::optional<PrefabSpaceLink> spaceLink;
};

struct PrefabStatic
{
    std::string mesh;
    int64_t x = 0, z = 0;
    int32_t y    = 0;
    uint16_t yaw = 0;
};

struct Prefab
{
    std::string name;
    uint16_t version = 1;
    std::string file;
    PrefabRect footprint;
    std::optional<PrefabPatch> patch;
    std::vector<PrefabRect> exclusion;
    std::vector<PrefabEntity> entities;
    std::vector<PrefabStatic> statics;
    std::vector<std::string> spaces;  // template names
    // FNV-1a over the integer content (not the component values); enters the plan hash
    uint64_t contentHash = 0;
};

// The file of a prefab version: <directory>/prefabs/<name>.json (1) or <name>.v<N>.json
SAIGA_ENGINE_API std::string prefabPath(const std::string& dataDirectory, const std::string& name, uint16_t version);
// Parses and checks a prefab (everything that needs no World); false with an error naming the file and the key
SAIGA_ENGINE_API bool parsePrefab(const json11::Json& json, const std::string& file, Prefab& out, std::string& error);
// Reads, parses and checks a prefab file; false with an error naming the file (a missing file, a JSON error, a key)
SAIGA_ENGINE_API bool readPrefab(const std::string& dataDirectory, const std::string& name, uint16_t version,
                                 Prefab& out, std::string& error);
SAIGA_ENGINE_API uint64_t prefabContentHash(const Prefab& p);
// A prefab yaw in degrees as the 65536-step yaw: floorDiv(deg * 65536 + 180, 360), wrapped
SAIGA_ENGINE_API uint16_t prefabYaw(int64_t degrees);

}  // namespace Engine
}  // namespace Saiga
