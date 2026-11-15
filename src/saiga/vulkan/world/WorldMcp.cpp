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

// yaw: 0 looks along -z, 90 along +x. pitch: positive looks up.
vec3 directionFromYawPitch(float yawDeg, float pitchDeg)
{
    float y = yawDeg * pi / 180.0f, p = pitchDeg * pi / 180.0f;
    return vec3(std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p));
}

void yawPitchFromDirection(const vec3& d, float& yawDeg, float& pitchDeg)
{
    yawDeg   = std::atan2(d(0), -d(2)) * 180.0f / pi;
    pitchDeg = std::asin(std::min(std::max(d(1), -1.0f), 1.0f)) * 180.0f / pi;
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

Json cameraJson(const WorldMcpContext& c)
{
    vec3 pos = c.camera->getPosition();
    vec3 fwd = cameraForward(*c.camera);
    float yaw, pitch;
    yawPitchFromDirection(fwd, yaw, pitch);
    float ground = c.world->heightField().height(pos(0), pos(2));
    Json::object o{{"position", toJson(pos)},
                   {"forward", toJson(fwd)},
                   {"yaw_deg", std::round(yaw * 10) / 10},
                   {"pitch_deg", std::round(pitch * 10) / 10},
                   {"fovy_deg", c.world->settings.fovy},
                   {"ground_height", std::round(ground * 100) / 100},
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

Json percentiles(std::vector<double> v)
{
    if (v.empty()) return Json();
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    auto at = [&](double q) { return v[std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5))]; };
    auto r  = [](double x) { return std::round(x * 1000) / 1000; };
    return Json::object{{"avg", r(sum / v.size())},
                        {"min", r(v.front())},
                        {"p50", r(at(0.5))},
                        {"p95", r(at(0.95))},
                        {"max", r(v.back())}};
}

// Integer box filter so that the width is <= maxWidth
TemplatedImage<ucvec3> downscale(const TemplatedImage<ucvec4>& img, int maxWidth)
{
    int f = std::max(1, (img.w + maxWidth - 1) / std::max(maxWidth, 1));
    int w = img.w / f, h = img.h / f;
    TemplatedImage<ucvec3> out(h, w);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            int s[3] = {0, 0, 0};
            for (int dy = 0; dy < f; ++dy)
                for (int dx = 0; dx < f; ++dx)
                    for (int k = 0; k < 3; ++k) s[k] += img(y * f + dy, x * f + dx)(k);
            for (int k = 0; k < 3; ++k) out(y, x)(k) = (unsigned char)(s[k] / (f * f));
        }
    }
    return out;
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

// An integer argument clamped to [lo, hi] (json11's int_value() is undefined for doubles outside the int range)
int intArgument(const Json& value, int lo, int hi, int defaultValue)
{
    if (!value.is_number()) return defaultValue;
    double v = value.number_value();
    if (!(v >= lo)) return lo;  // also NaN
    if (v > hi) return hi;
    return int(v);
}

// Several screenshot requests before the next frame share it if they want the same GUI option: the GUI flag is saved
// by the first and restored by the last one. A request with the other option waits for the next batch (frame).
struct ScreenshotBatch
{
    int pending      = 0;
    bool gui         = false;
    bool previousGui = false;
    std::vector<std::function<void()>> deferred;
};

using ScreenshotHandler = std::function<void(const TemplatedImage<ucvec4>&)>;

void requestScreenshot(const WorldMcpContext& c, std::shared_ptr<ScreenshotBatch> batch, bool gui,
                       ScreenshotHandler handler)
{
    if (batch->pending > 0 && batch->gui != gui)
    {
        batch->deferred.push_back([c, batch, gui, handler]() { requestScreenshot(c, batch, gui, handler); });
        return;
    }
    if (batch->pending++ == 0)
    {
        batch->gui                      = gui;
        batch->previousGui              = c.renderer->should_render_imgui;
        c.renderer->should_render_imgui = gui;
    }
    c.renderer->requestScreenshot(
        [c, batch, handler](const TemplatedImage<ucvec4>& img)
        {
            if (--batch->pending == 0)
            {
                c.renderer->should_render_imgui = batch->previousGui;
                // The waiting requests form the next batch (the callbacks run after the frame, so they are rendered in
                // the next one)
                auto deferred = std::move(batch->deferred);
                batch->deferred.clear();
                for (auto& f : deferred) f();
            }
            handler(img);
        });
}

}  // namespace

std::string worldMcpInstructions()
{
    return "Controls a running Saiga Vulkan world renderer (procedural terrain, trees and grass in a gray clay look). "
           "Coordinates are meters, y is up, the world covers x and z in [0, size_m]. Call get_state first. "
           "Use screenshot to look at the current view (the image is the next rendered frame). set_camera places the "
           "camera (position / look_at / yaw_deg / pitch_deg / height_above_ground), move_camera moves it relative to "
           "the view. set_settings changes render settings (get_state lists them). measure_performance renders "
           "frames and reports frame times; it is only meaningful with vsync off and the validation layer disabled.";
}

void registerWorldMcpTools(McpServer& server, const WorldMcpContext& context)
{
    SAIGA_ASSERT(context.renderer && context.world && context.camera);
    WorldMcpContext c = context;
    auto batch        = std::make_shared<ScreenshotBatch>();
    // measure_performance answers before the tool call times out
    double timeoutSeconds = server.toolTimeoutSeconds;

    server.addTool(
        "screenshot",
        "Renders the next frame and returns it as an image, followed by the camera state. The GUI is hidden unless "
        "gui=true.",
        McpServer::ObjectSchema(
            {{"gui", McpServer::BooleanProperty("Include the ImGui windows (default false)")},
             {"format", McpServer::StringProperty("Image format (default jpg)", {"jpg", "png"})},
             {"max_width", McpServer::IntegerProperty("Downscale to at most this width (default 1280)")}}),
        [c, batch](const Json& args, McpServer::Respond respond)
        {
            bool gui           = args["gui"].bool_value();
            std::string format = args["format"].is_string() ? args["format"].string_value() : "jpg";
            int maxWidth       = intArgument(args["max_width"], 64, 8192, 1280);
            if (format != "png") format = "jpg";

            requestScreenshot(
                c, batch, gui,
                [c, respond, format, maxWidth](const TemplatedImage<ucvec4>& img)
                {
                    if (img.width == 0)
                    {
                        respond(ToolResult::Error("The frame could not be read back (see the application output)"));
                        return;
                    }
                    auto small = downscale(img, maxWidth);
                    small.set_compression_quality(85);
                    auto data = small.saveToMemory(format);
                    if (data.empty())
                    {
                        respond(ToolResult::Error("Could not encode the screenshot as " + format));
                        return;
                    }
                    ToolResult r = ToolResult::Image(data, format == "png" ? "image/png" : "image/jpeg");
                    r.addText(McpServer::FormatJson(cameraJson(c), false));
                    respond(r);
                });
        });

    server.addTool("get_state",
                   "Returns the camera, the world extent, render statistics (fps, visible trees, triangles) and all "
                   "render settings.",
                   McpServer::ObjectSchema({}),
                   [c](const Json&, McpServer::Respond respond)
                   {
                       respond(ToolResult::FromJson(Json::object{{"camera", cameraJson(c)},
                                                                 {"world", worldJson(c)},
                                                                 {"stats", c.world->statsJson()},
                                                                 {"settings", c.world->settings.toJson()}}));
                   });

    server.addTool(
        "set_camera",
        "Places the camera. All arguments are optional: position [x,y,z]; height_above_ground overrides the y of the "
        "position with the terrain height + this value; the view direction from look_at [x,y,z], or from yaw_deg "
        "(0 looks along -z, 90 along +x) and pitch_deg (positive looks up), unchanged otherwise. walk_mode keeps the "
        "camera at eye height above the terrain.",
        McpServer::ObjectSchema(
            {{"position", McpServer::Vec3Property("Camera position [x, y, z] in meters")},
             {"look_at", McpServer::Vec3Property("Point to look at [x, y, z]")},
             {"yaw_deg", McpServer::NumberProperty("Heading in degrees, 0 = -z, 90 = +x")},
             {"pitch_deg", McpServer::NumberProperty("Pitch in degrees, positive looks up")},
             {"height_above_ground", McpServer::NumberProperty("Height above the terrain in meters")},
             {"fovy_deg", McpServer::NumberProperty("Vertical field of view in degrees")},
             {"walk_mode", McpServer::BooleanProperty("Walk on the terrain")}}),
        [c](const Json& args, McpServer::Respond respond)
        {
            const double limit = maxWorldSizes * c.world->heightField().worldSize();
            vec3 pos           = c.camera->getPosition();
            vec3 dir           = cameraForward(*c.camera);
            if (!args["position"].is_null() && !fromJson(args["position"], pos, limit))
            {
                respond(ToolResult::Error("position must be [x, y, z] with finite numbers"));
                return;
            }
            float heightAboveGround = 0, yaw = 0, pitch = 0, fovy = c.world->settings.fovy;
            if (!optionalNumber(args, "height_above_ground", limit, heightAboveGround, respond) ||
                !optionalNumber(args, "yaw_deg", 1e6, yaw, respond) ||
                !optionalNumber(args, "pitch_deg", 1e6, pitch, respond) ||
                !optionalNumber(args, "fovy_deg", 1e6, fovy, respond))
            {
                return;
            }
            if (args["height_above_ground"].is_number())
            {
                pos(1) = c.world->heightField().height(pos(0), pos(2)) + heightAboveGround;
            }
            if (args["walk_mode"].is_bool() && c.walkMode)
            {
                *c.walkMode = args["walk_mode"].bool_value();
                if (*c.walkMode && c.eyeHeight) pos(1) = c.world->heightField().height(pos(0), pos(2)) + *c.eyeHeight;
            }
            if (args["look_at"].is_array())
            {
                vec3 target;
                if (!fromJson(args["look_at"], target, limit))
                {
                    respond(ToolResult::Error("look_at must be [x, y, z] with finite numbers"));
                    return;
                }
                dir = target - pos;
            }
            else if (args["yaw_deg"].is_number() || args["pitch_deg"].is_number())
            {
                float currentYaw, currentPitch;
                yawPitchFromDirection(dir, currentYaw, currentPitch);
                dir = directionFromYawPitch(args["yaw_deg"].is_number() ? yaw : currentYaw,
                                            args["pitch_deg"].is_number() ? pitch : currentPitch);
            }
            if (args["fovy_deg"].is_number())
            {
                // The same range as the settings (GUI and set_settings)
                c.world->settings.fovy = fovy;
                c.world->settings.clamp();
            }
            lookFrom(*c.camera, pos, dir);
            respond(ToolResult::FromJson(cameraJson(c)));
        });

    server.addTool(
        "move_camera",
        "Moves the camera relative to the view: forward/right in meters along the ground plane, up in meters, then "
        "turns by yaw_deg (positive: to the right) and pitch_deg (positive: up).",
        McpServer::ObjectSchema({{"forward", McpServer::NumberProperty("Meters forward (horizontal)")},
                                 {"right", McpServer::NumberProperty("Meters to the right")},
                                 {"up", McpServer::NumberProperty("Meters up")},
                                 {"yaw_deg", McpServer::NumberProperty("Turn right by this angle")},
                                 {"pitch_deg", McpServer::NumberProperty("Turn up by this angle")}}),
        [c](const Json& args, McpServer::Respond respond)
        {
            const double limit = maxWorldSizes * c.world->heightField().worldSize();
            float forward = 0, rightOffset = 0, up = 0, yawDelta = 0, pitchDelta = 0;
            if (!optionalNumber(args, "forward", limit, forward, respond) ||
                !optionalNumber(args, "right", limit, rightOffset, respond) ||
                !optionalNumber(args, "up", limit, up, respond) ||
                !optionalNumber(args, "yaw_deg", 1e6, yawDelta, respond) ||
                !optionalNumber(args, "pitch_deg", 1e6, pitchDelta, respond))
            {
                return;
            }
            vec3 pos = c.camera->getPosition();
            vec3 dir = cameraForward(*c.camera);
            float yaw, pitch;
            yawPitchFromDirection(dir, yaw, pitch);
            vec3 flatForward = directionFromYawPitch(yaw, 0);
            vec3 right       = directionFromYawPitch(yaw + 90, 0);
            pos += flatForward * forward + right * rightOffset + vec3(0, 1, 0) * up;
            for (int i = 0; i < 3; ++i) pos(i) = float(std::min(std::max(double(pos(i)), -limit), limit));
            yaw += yawDelta;
            pitch = std::min(std::max(pitch + pitchDelta, -89.0f), 89.0f);
            lookFrom(*c.camera, pos, directionFromYawPitch(yaw, pitch));
            respond(ToolResult::FromJson(cameraJson(c)));
        });

    server.addTool("set_settings",
                   "Changes render settings (only the given ones; values are clamped to the GUI ranges). Returns all "
                   "settings and invalid_keys (unknown keys or values of the wrong type, not applied). debug_mode: 0 "
                   "off, 1 tint by level of detail.",
                   settingsSchema(c.world->settings),
                   [c](const Json& args, McpServer::Respond respond)
                   {
                       auto unknown = c.world->settings.fromJson(args);
                       Json::object o{{"settings", c.world->settings.toJson()}};
                       if (!unknown.empty()) o["invalid_keys"] = Json::array(unknown.begin(), unknown.end());
                       respond(ToolResult::FromJson(o));
                   });

    server.addTool(
        "terrain_query",
        "Terrain height, normal and slope at world positions. points: [[x, z], ...] (at most 1000 points).",
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

    server.addTool(
        "measure_performance",
        "Renders the given number of frames (default 120) without changing anything and reports the CPU frame time "
        "and the GPU time of the world passes in ms (avg, min, p50, p95, max) and the render statistics. At low frame "
        "rates it stops early (before the tool call times out) and reports the frames measured so far.",
        McpServer::ObjectSchema({{"frames", McpServer::IntegerProperty("Number of frames (1..5000)")}}),
        [c, timeoutSeconds](const Json& args, McpServer::Respond respond)
        {
            int frames    = intArgument(args["frames"], 1, 5000, 120);
            auto cpu      = std::make_shared<std::vector<double>>();
            auto gpu      = std::make_shared<std::vector<double>>();
            auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(0.8 * timeoutSeconds);
            // Only frames prepared after this request: the frame prepared last may still use the previous settings
            // (e.g. set_settings right before), and its GPU times arrive a few frames later
            int firstFrame = c.world->frameNumber() + 1;
            c.world->frameCallbacks.push_back(
                [c, frames, cpu, gpu, respond, deadline, firstFrame](const WorldRenderer::FrameSample& s)
                {
                    if (s.cpuFrame >= firstFrame) cpu->push_back(s.cpuMs);
                    if (s.gpuMs >= 0 && s.gpuFrame >= firstFrame) gpu->push_back(s.gpuMs);
                    if ((int)cpu->size() < frames && std::chrono::steady_clock::now() < deadline) return false;
                    respond(ToolResult::FromJson(Json::object{{"frames", (int)cpu->size()},
                                                              {"requested_frames", frames},
                                                              {"cpu_frame_ms", percentiles(*cpu)},
                                                              {"gpu_world_ms", percentiles(*gpu)},
                                                              {"stats", c.world->statsJson()},
                                                              {"camera", cameraJson(c)}}));
                    return true;
                });
        });
}

}  // namespace Vulkan
}  // namespace Saiga
