/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/WorldMcp.h"

#include "saiga/core/image/managedImage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

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

bool fromJson(const Json& j, vec3& v, double limit)
{
    if (!j.is_array() || j.array_items().size() != 3) return false;
    for (int i = 0; i < 3; ++i)
    {
        if (!finiteClamped(j[i], limit, v(i))) return false;
    }
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

vec3 cameraForward(const Camera& camera)
{
    return -make_vec3(camera.model.col(2)).normalized();
}

void lookFrom(Camera& camera, const vec3& eye, vec3 dir)
{
    if (!dir.allFinite() || dir.norm() < 1e-6f) dir = vec3(0, 0, -1);
    dir.normalize();
    // Keep away from straight up/down, the view matrix needs a non parallel up vector. The horizontal part is rebuilt
    // from its direction (adding a small offset could cancel a small existing component, e.g. x = -0.001), with the
    // horizontal direction of the current view as the fallback when it is (almost) zero.
    if (std::abs(dir(1)) > 0.999f)
    {
        vec2 h(dir(0), dir(2));
        if (h.norm() < 1e-4f)
        {
            vec3 f = cameraForward(camera);
            h      = vec2(f(0), f(2));
            if (!h.allFinite() || h.norm() < 1e-4f) h = vec2(0, 1);
        }
        h          = h.normalized() * std::sqrt(1.0f - 0.999f * 0.999f);
        float sign = dir(1) < 0 ? -1.0f : 1.0f;
        dir        = vec3(h(0), 0.999f * sign, h(1));
    }
    // From the direction, not from eye + dir: far from the origin (and for steep directions) eye + dir loses the small
    // components of dir in float
    camera.setView(lookAt(vec3(0, 0, 0), dir, vec3(0, 1, 0)) * translate(-eye));
}

// The camera's ground and walk-mode state (the rest of the camera is in the app's app_status)
Json groundJson(const WorldMcpContext& c)
{
    const vec3 pos     = c.camera->getPosition();
    const float ground = c.world->heightField().height(pos(0), pos(2));
    Json::object o{{"ground_height", std::round(ground * 100) / 100},
                   {"height_above_ground", std::round((pos(1) - ground) * 100) / 100}};
    if (c.walkMode) o["walk_mode"] = *c.walkMode;
    return o;
}

Json worldJson(const WorldMcpContext& c)
{
    auto& hf = c.world->heightField();
    return Json::object{
        {"size_m", hf.worldSize()},
        {"extent", "x and z in [0, size_m], y is up"},
        {"min_height", hf.minHeight()},
        {"max_height", hf.maxHeight()},
        {"water_level", hf.waterLevel()},
        {"highest_point", toJson(hf.highestPoint())},
        {"center", toJson(vec3(hf.worldSize() * 0.5f, hf.height(hf.worldSize() * 0.5f, hf.worldSize() * 0.5f),
                               hf.worldSize() * 0.5f))}};
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
    SAIGA_ASSERT(context.world && context.camera);
    WorldMcpContext c = context;

    server.addTool("game_get_state",
                   "Reads the world extent, render statistics (fps, visible trees, triangles), all render settings and "
                   "the camera's ground height, height above ground and walk mode (the camera itself: app_status). "
                   "Changes nothing.",
                   McpServer::ObjectSchema({}),
                   [c](const Json&, McpServer::Respond respond)
                   {
                       respond(ToolResult::FromJson(Json::object{{"camera", groundJson(c)},
                                                                 {"world", worldJson(c)},
                                                                 {"stats", c.world->statsJson()},
                                                                 {"settings", c.world->settings.toJson()}}));
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
            const double limit = maxWorldSizes * c.world->heightField().worldSize();
            vec3 pos           = c.camera->getPosition();
            const vec3 dir     = cameraForward(*c.camera);
            if (!args["position"].is_null())
            {
                const Json& p  = args["position"];
                const size_t n = p.is_array() ? p.array_items().size() : 0;
                float x = 0, y = pos(1), z = 0;
                const bool ok = (n == 3 && finiteClamped(p[0], limit, x) && finiteClamped(p[1], limit, y) &&
                                 finiteClamped(p[2], limit, z)) ||
                                (n == 2 && finiteClamped(p[0], limit, x) && finiteClamped(p[1], limit, z));
                if (!ok) return respond(ToolResult::Error("position: [x, y, z] or [x, z] with finite numbers"));
                pos = vec3(x, y, z);
            }
            float heightAboveGround = 0;
            if (!optionalNumber(args, "height_above_ground", limit, heightAboveGround, respond)) return;
            if (!args["walk_mode"].is_null() && !args["walk_mode"].is_bool())
                return respond(ToolResult::Error("walk_mode: a bool"));
            if (args["height_above_ground"].is_number())
                pos(1) = c.world->heightField().height(pos(0), pos(2)) + heightAboveGround;
            if (args["walk_mode"].is_bool() && c.walkMode)
            {
                *c.walkMode = args["walk_mode"].bool_value();
                if (*c.walkMode && c.eyeHeight) pos(1) = c.world->heightField().height(pos(0), pos(2)) + *c.eyeHeight;
            }
            lookFrom(*c.camera, pos, dir);
            Json::object o = groundJson(c).object_items();
            o["position"]  = toJson(c.camera->getPosition());
            respond(ToolResult::FromJson(o));
        });

    server.addTool("game_set_settings",
                   "Changes render settings of the prototype (only the given ones; values are clamped to the GUI "
                   "ranges). Returns all settings and invalid_keys (unknown keys or values of the wrong type, not "
                   "applied). debug_mode: 0 off, 1 tint by level of detail. Changes the render settings.",
                   settingsSchema(c.world->settings),
                   [c](const Json& args, McpServer::Respond respond)
                   {
                       auto unknown = c.world->settings.fromJson(args);
                       Json::object o{{"settings", c.world->settings.toJson()}};
                       if (!unknown.empty()) o["invalid_keys"] = Json::array(unknown.begin(), unknown.end());
                       respond(ToolResult::FromJson(o));
                   });

    server.addTool(
        "game_terrain_query",
        "Terrain height, normal and slope of the prototype height field at world positions. points: [[x, z], ...] (at "
        "most 1000 points). Reads; changes nothing.",
        McpServer::ObjectSchema(
            {{"points",
              Json::object{{"type", "array"},
                           {"items", Json::object{{"type", "array"}, {"items", Json::object{{"type", "number"}}}}},
                           {"description", "List of [x, z]"}}}},
            {"points"}),
        [c](const Json& args, McpServer::Respond respond)
        {
            auto& hf           = c.world->heightField();
            const double limit = maxWorldSizes * hf.worldSize();
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
                vec3 n = hf.normal(x, z, 2.0f);
                result.push_back(
                    Json::object{{"x", x},
                                 {"z", z},
                                 {"height", std::round(hf.height(x, z) * 100) / 100},
                                 {"normal", toJson(n)},
                                 {"slope_deg", std::round(std::acos(std::min(n(1), 1.0f)) * 180.0f / pi * 10) / 10},
                                 {"under_water", hf.height(x, z) < hf.waterLevel()}});
            }
            respond(ToolResult::FromJson(result));
        });
}

}  // namespace Vulkan
}  // namespace Saiga
