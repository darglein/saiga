/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/WorldMcp.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
using Json       = McpServer::Json;
using ToolResult = McpServer::ToolResult;

namespace
{
constexpr float pi = 3.14159265358979f;

Json toJson(const vec3& v)
{
    // Rounded to cm, enough for the agent
    auto r = [](float x) { return std::round(x * 100.0) / 100.0; };
    return Json::array{r(v(0)), r(v(1)), r(v(2))};
}

// Camera coordinates and offsets are clamped to this range (multiple of the world size): beyond it the float math of
// the renderer (grass cells, view matrix) breaks down.
constexpr double maxWorldSizes = 10;

// A finite number clamped to [-limit, limit]. Returns false for non-finite values.
bool finiteClamped(const Json& j, double limit, float& out)
{
    if (!j.is_number() || !std::isfinite(j.number_value())) return false;
    out = float(std::min(std::max(j.number_value(), -limit), limit));
    return true;
}

// Reads the optional number argument name into out. Returns false (and responds with an error) if it is present but
// not a finite number.
bool optionalNumber(const Json& args, const char* name, double limit, float& out, const McpServer::Respond& respond)
{
    const Json& j = args[name];
    if (j.is_null()) return true;
    if (finiteClamped(j, limit, out)) return true;
    respond(ToolResult::Error(std::string(name) + " must be a finite number"));
    return false;
}

// The camera's ground and walk-mode state (the rest of the camera is in the app's app_status)
Json groundJson(const WorldMcpContext& c)
{
    const Vec3 pos      = c.getCamera().position;
    const double ground = c.ground(pos(0), pos(2));
    Json::object o{{"ground_height", std::round(ground * 100) / 100},
                   {"height_above_ground", std::round((pos(1) - ground) * 100) / 100}};
    if (c.walkMode) o["walk_mode"] = *c.walkMode;
    return o;
}

Json worldJson(const WorldMcpContext& c)
{
    const auto range = c.heightRange();
    const double mid = c.worldSize * 0.5;
    return Json::object{{"size_m", c.worldSize},
                        {"extent", "x and z in [0, size_m], y is up"},
                        {"min_height", range.first},
                        {"max_height", range.second},
                        {"water_level", c.seaLevel},
                        {"highest_point", toJson(c.highestPoint.cast<float>())},
                        {"center", toJson(vec3(float(mid), float(c.ground(mid, mid)), float(mid)))}};
}

// The normal of the ground by central differences at step metres
vec3 groundNormal(const WorldMcpContext& c, double x, double z, double step)
{
    const double hl = c.ground(x - step, z), hr = c.ground(x + step, z);
    const double hd = c.ground(x, z - step), hu = c.ground(x, z + step);
    return Vec3(hl - hr, 2.0 * step, hd - hu).normalized().cast<float>();
}

Json settingsSchema(const WorldSettings& settings)
{
    std::map<std::string, Json> props;
    Json current = settings.toJson();
    for (auto& kv : current.object_items())
    {
        std::string description = "current: " + kv.second.dump();
        if (kv.first == "shading")
            props[kv.first] = McpServer::StringProperty(description, {"clay"});
        else if (kv.first == "debug_mode")
            props[kv.first] = McpServer::IntegerProperty(description + " (0: off, 1: tint by level of detail)");
        else
            props[kv.first] =
                Json::object{{"type", kv.second.is_bool() ? "boolean" : "number"}, {"description", description}};
    }
    return McpServer::ObjectSchema(props);
}

}  // namespace

std::string worldMcpInstructions()
{
    return "Controls a running engine game on the Saiga world prototype (procedural terrain, trees and grass in a gray "
           "clay look). Coordinates are meters, y is up, the world covers x and z in [0, size_m]. Call app_status and "
           "game_get_state first. screenshot shows the next frame; set_camera / move_camera place the app camera "
           "(game_set_camera: height above the terrain, walk mode); game_set_settings changes render settings "
           "(game_get_state lists them), game_terrain_query reads the terrain. The world API tools (describe_world, "
           "list_entities, inspect, step, save_snapshot, ...) read and change the game world; measure_performance "
           "measures steps, or frames and the GPU (vsync off for frame times); list_scenarios / run_scenario run the "
           "game's scenarios.";
}

void registerWorldMcpTools(McpServer& server, const WorldMcpContext& context)
{
    SAIGA_ASSERT(context.getCamera && context.setCamera);
    WorldMcpContext c = context;
    if (c.world)
    {
        // The S19a path: the fields from the WorldRenderer and its height field
        WorldRenderer* w = c.world;
        c.settings       = &w->settings;
        c.stats          = [w]() { return w->statsJson(); };
        c.ground         = [w](double x, double z) { return double(w->heightField().height(float(x), float(z))); };
        c.worldSize      = w->heightField().worldSize();
        c.seaLevel       = w->heightField().waterLevel();
        c.highestPoint   = w->heightField().highestPoint().cast<double>();
        c.heightRange    = [w]()
        { return std::make_pair(double(w->heightField().minHeight()), double(w->heightField().maxHeight())); };
    }
    SAIGA_ASSERT(c.settings && c.stats && c.ground && c.heightRange && c.worldSize > 0,
                 "registerWorldMcpTools: the context's settings, stats, ground, heightRange and worldSize");

    server.addTool("game_get_state",
                   "Reads the world extent, render statistics (fps, visible trees, triangles), all render settings and "
                   "the camera's ground height, height above ground and walk mode (the camera itself: app_status). "
                   "Changes nothing.",
                   McpServer::ObjectSchema({}),
                   [c](const Json&, McpServer::Respond respond)
                   {
                       respond(ToolResult::FromJson(Json::object{{"camera", groundJson(c)},
                                                                 {"world", worldJson(c)},
                                                                 {"stats", c.stats()},
                                                                 {"settings", c.settings->toJson()}}));
                   });

    server.addTool(
        "game_set_camera",
        "The camera's height and walk mode on the prototype terrain: position [x, y, z] or [x, z] (moves the camera, "
        "keeps the view direction), height_above_ground sets y to the terrain height plus the value, walk_mode keeps "
        "the camera at eye height above the terrain. Returns the camera position, ground_height and "
        "height_above_ground. Changes the camera (no camera hold: the app camera is this game's render camera).",
        McpServer::ObjectSchema(
            {{"position", Json::object{{"type", "array"},
                                       {"items", Json::object{{"type", "number"}}},
                                       {"description", "[x, y, z] or [x, z] in meters"}}},
             {"height_above_ground", McpServer::NumberProperty("Height above the terrain in meters")},
             {"walk_mode", McpServer::BooleanProperty("Walk on the terrain")}}),
        [c](const Json& args, McpServer::Respond respond)
        {
            for (auto& kv : args.object_items())
                if (kv.first != "position" && kv.first != "height_above_ground" && kv.first != "walk_mode")
                    return respond(ToolResult::Error("unknown argument \"" + kv.first + "\""));
            const double limit = maxWorldSizes * c.worldSize;
            WorldMcpCamera cam = c.getCamera();
            Vec3 pos           = cam.position;
            if (!args["position"].is_null())
            {
                const Json& p  = args["position"];
                const size_t n = p.is_array() ? p.array_items().size() : 0;
                float x = 0, y = float(pos(1)), z = 0;
                const bool ok = (n == 3 && finiteClamped(p[0], limit, x) && finiteClamped(p[1], limit, y) &&
                                 finiteClamped(p[2], limit, z)) ||
                                (n == 2 && finiteClamped(p[0], limit, x) && finiteClamped(p[1], limit, z));
                if (!ok) return respond(ToolResult::Error("position: [x, y, z] or [x, z] with finite numbers"));
                pos = Vec3(x, y, z);
            }
            float heightAboveGround = 0;
            if (!optionalNumber(args, "height_above_ground", limit, heightAboveGround, respond)) return;
            if (!args["walk_mode"].is_null() && !args["walk_mode"].is_bool())
                return respond(ToolResult::Error("walk_mode: a bool"));
            if (args["height_above_ground"].is_number()) pos(1) = c.ground(pos(0), pos(2)) + heightAboveGround;
            if (args["walk_mode"].is_bool() && c.walkMode)
            {
                *c.walkMode = args["walk_mode"].bool_value();
                if (*c.walkMode && c.eyeHeight) pos(1) = c.ground(pos(0), pos(2)) + *c.eyeHeight;
            }
            // The view direction stays
            cam.position = pos;
            c.setCamera(cam);
            Json::object o = groundJson(c).object_items();
            o["position"]  = toJson(vec3(c.getCamera().position.cast<float>()));
            respond(ToolResult::FromJson(o));
        });

    server.addTool("game_set_settings",
                   "Changes render settings of the prototype (only the given ones; values are clamped to the GUI "
                   "ranges). Returns all settings and invalid_keys (unknown keys or values of the wrong type, not "
                   "applied). debug_mode: 0 off, 1 tint by level of detail. Changes the render settings.",
                   settingsSchema(*c.settings),
                   [c](const Json& args, McpServer::Respond respond)
                   {
                       auto unknown = c.settings->fromJson(args);
                       // The field of view is the app camera's (S19b)
                       if (args["fovy"].is_number())
                       {
                           WorldMcpCamera cam = c.getCamera();
                           cam.fovy           = radians(double(c.settings->fovy));
                           c.setCamera(cam);
                       }
                       Json::object o{{"settings", c.settings->toJson()}};
                       if (!unknown.empty()) o["invalid_keys"] = Json::array(unknown.begin(), unknown.end());
                       respond(ToolResult::FromJson(o));
                   });

    server.addTool(
        "game_terrain_query",
        "Terrain height, normal (central differences at 2 m) and slope of the terrain at world positions. points: [[x, "
        "z], ...] (at "
        "most 1000 points). Reads; changes nothing.",
        McpServer::ObjectSchema(
            {{"points",
              Json::object{{"type", "array"},
                           {"items", Json::object{{"type", "array"}, {"items", Json::object{{"type", "number"}}}}},
                           {"description", "List of [x, z]"}}}},
            {"points"}),
        [c](const Json& args, McpServer::Respond respond)
        {
            const double limit = maxWorldSizes * c.worldSize;
            const auto& points = args["points"];
            if (!points.is_array() || points.array_items().size() > 1000)
            {
                respond(ToolResult::Error("points must be an array of at most 1000 [x, z] pairs"));
                return;
            }
            Json::array result;
            for (auto& p : points.array_items())
            {
                float x, z;
                if (!p.is_array() || p.array_items().size() != 2 || !finiteClamped(p[0], limit, x) ||
                    !finiteClamped(p[1], limit, z))
                {
                    respond(ToolResult::Error("every point must be [x, z] with finite numbers"));
                    return;
                }
                const vec3 n   = groundNormal(c, x, z, 2.0);
                const double h = c.ground(x, z);
                result.push_back(
                    Json::object{{"x", x},
                                 {"z", z},
                                 {"height", std::round(h * 100) / 100},
                                 {"normal", toJson(n)},
                                 {"slope_deg", std::round(std::acos(std::min(n(1), 1.0f)) * 180.0f / pi * 10) / 10},
                                 {"under_water", h < c.seaLevel}});
            }
            respond(ToolResult::FromJson(result));
        });
}

}  // namespace Vulkan
}  // namespace Saiga
