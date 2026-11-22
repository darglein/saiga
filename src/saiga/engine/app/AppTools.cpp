/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The app's MCP tools: quit, app_status, set_input (S13); screenshot, set_camera, move_camera, list_scenarios,
// run_scenario (S14).

#include "saiga/engine/app/AppTools.h"

#include "saiga/core/image/managedImage.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/app/LightingTools.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioRunner.h"
#include "saiga/engine/tools/WorldTools.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>

namespace Saiga
{
namespace Engine
{
namespace
{
using Json       = json11::Json;
using ToolResult = McpServer::ToolResult;

Json schema(const Json::object& properties)
{
    return Json::object{{"type", "object"}, {"properties", properties}, {"additionalProperties", false}};
}

constexpr double kPi = 3.14159265358979323846;
// Camera coordinates: finite and within this range (the app camera is float: cm resolution there)
constexpr double kCameraLimit = 1e6;

bool unknownArgument(const Json& args, std::initializer_list<const char*> known, const McpServer::Respond& respond)
{
    for (auto& kv : args.object_items())
    {
        bool ok = false;
        for (const char* k : known) ok = ok || kv.first == k;
        if (!ok)
        {
            respond(ToolResult::Error("unknown argument \"" + kv.first + "\""));
            return true;
        }
    }
    return false;
}

// A finite number within limit (absent: keeps out); false and an error answer otherwise
bool numberArgument(const Json& args, const char* name, double limit, double& out, bool& given,
                    const McpServer::Respond& respond)
{
    const Json& j = args[name];
    given         = !j.is_null();
    if (!given) return true;
    if (!j.is_number() || !std::isfinite(j.number_value()) || std::abs(j.number_value()) > limit)
    {
        respond(ToolResult::Error(std::string(name) + ": a finite number within " + std::to_string(int64_t(limit))));
        return false;
    }
    out = j.number_value();
    return true;
}

bool vec3Argument(const Json& args, const char* name, Vec3& out, bool& given, const McpServer::Respond& respond)
{
    const Json& j = args[name];
    given         = !j.is_null();
    if (!given) return true;
    bool ok = j.is_array() && j.array_items().size() == 3;
    for (int i = 0; ok && i < 3; ++i)
    {
        ok = j[size_t(i)].is_number() && std::isfinite(j[size_t(i)].number_value()) &&
             std::abs(j[size_t(i)].number_value()) <= kCameraLimit;
        if (ok) out(i) = j[size_t(i)].number_value();
    }
    if (!ok) respond(ToolResult::Error(std::string(name) + ": [x, y, z] with finite numbers within 1e6"));
    return ok;
}

// The view camera's forward direction (-z of its rotation)
Vec3 cameraForward(const ViewCamera& camera)
{
    return camera.rotation * Vec3(0, 0, -1);
}

// yaw: 0 looks along -z, 90 along +x; pitch: positive looks up
Vec3 directionFromYawPitch(double yawDeg, double pitchDeg)
{
    const double y = yawDeg * kPi / 180.0, p = pitchDeg * kPi / 180.0;
    return Vec3(std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p));
}

void yawPitchFromDirection(const Vec3& d, double& yawDeg, double& pitchDeg)
{
    yawDeg   = std::atan2(d(0), -d(2)) * 180.0 / kPi;
    pitchDeg = std::asin(std::min(std::max(d(1), -1.0), 1.0)) * 180.0 / kPi;
}

// A tool's camera write (held): the view camera at eye looking along dir (a heading kept near straight up / down)
void lookFrom(EngineApp& app, const Vec3& eye, const Vec3& dir, std::optional<SpaceId> space = std::nullopt)
{
    const ViewCamera& c = app.viewCamera();
    app.setViewCameraByTool(space.value_or(c.space), eye, cameraLookRotation(dir, cameraForward(c)), c.fovy, true);
}

double round3(double x)
{
    return std::round(x * 1000) / 1000;
}

Json cameraJson(EngineApp& app)
{
    const ViewCamera& c = app.viewCamera();
    const Vec3& p       = c.position;
    const Vec3 f        = cameraForward(c);
    double yaw, pitch;
    yawPitchFromDirection(f, yaw, pitch);
    return Json::object{{"space", int(c.space)},
                        {"position", Json::array{round3(p(0)), round3(p(1)), round3(p(2))}},
                        {"forward", Json::array{round3(f(0)), round3(f(1)), round3(f(2))}},
                        {"yaw_deg", std::round(yaw * 10) / 10},
                        {"pitch_deg", std::round(pitch * 10) / 10},
                        {"held", app.cameraHeldByTool()}};
}

// Box filter to exactly maxWidth when wider (the height keeps the aspect ratio; never upscaled)
TemplatedImage<ucvec3> downscale(const TemplatedImage<ucvec4>& img, int maxWidth)
{
    const int w = std::min(img.w, maxWidth);
    const int h = std::max(1, int(std::lround(double(img.h) * w / std::max(1, img.w))));
    TemplatedImage<ucvec3> out(h, w);
    const double sx = double(img.w) / w, sy = double(img.h) / h;
    for (int y = 0; y < h; ++y)
    {
        const int y0 = int(y * sy), y1 = std::max(y0 + 1, std::min(img.h, int(std::ceil((y + 1) * sy))));
        for (int x = 0; x < w; ++x)
        {
            const int x0 = int(x * sx), x1 = std::max(x0 + 1, std::min(img.w, int(std::ceil((x + 1) * sx))));
            int sum[3] = {0, 0, 0};
            for (int yy = y0; yy < y1; ++yy)
                for (int xx = x0; xx < x1; ++xx)
                    for (int k = 0; k < 3; ++k) sum[k] += img(yy, xx)(k);
            const int n = (y1 - y0) * (x1 - x0);
            for (int k = 0; k < 3; ++k) out(y, x)(k) = (unsigned char)(sum[k] / n);
        }
    }
    return out;
}

// One screenshot request: answered exactly once (the frame or the deadline), then its answer (with the responder) is
// released
struct ScreenshotRequest
{
    bool gui = false;
    // S36a1b: the frame's lighting debug view and overlays (a one-frame override; nullopt: the user's)
    std::optional<Vulkan::LightingDebugState> debug;
    bool sameFrame(const ScreenshotRequest& o) const { return gui == o.gui && debug == o.debug; }
    bool answered = false;
    std::function<void(const TemplatedImage<ucvec4>&)> answer;
    // The answer deadline of the tool call (answerDeadline, S20): the frame's terrain completion is bounded by the
    // earliest pending one; an expired request is answered with an error (fail)
    std::chrono::steady_clock::time_point deadline;
    std::function<void(const std::string&)> fail;
};

// The screenshot requests before the next frame share one readback of it if they want the same GUI option and lighting
// debug view (drawn through RenderScene's ImGui and lighting debug overrides, so the user's choices are untouched); a
// request with another option waits for the next readback. At most one readback and one frame callback are registered
// (a readback cannot be cancelled: with nothing rendered it stays registered, but holds no request after the deadline
// answered them).
struct ScreenshotBatch
{
    bool inFlight = false;  // a readback and its frame callback are registered
    bool gui      = false;  // the option of current
    std::optional<Vulkan::LightingDebugState> debug;
    std::vector<std::shared_ptr<ScreenshotRequest>> current, deferred;
    bool hasImage = false;
    TemplatedImage<ucvec4> image;
};

void startScreenshot(EngineApp& app, const std::shared_ptr<ScreenshotBatch>& batch,
                     const std::shared_ptr<ScreenshotRequest>& r);

// After requests left the batch: the GUI option of the waiting ones (the deferred become current when none is left)
void updateBatch(EngineApp& app, ScreenshotBatch& batch)
{
    if (batch.current.empty() && !batch.deferred.empty())
    {
        batch.current.swap(batch.deferred);
        batch.gui   = batch.current.front()->gui;
        batch.debug = batch.current.front()->debug;
    }
    if (!app.scene()) return;
    if (batch.inFlight && !batch.current.empty())
    {
        app.scene()->setImguiOverride(batch.gui);
        app.scene()->setLightingDebugOverride(batch.debug);
    }
    else
    {
        app.scene()->setImguiOverride(std::nullopt);
        app.scene()->setLightingDebugOverride(std::nullopt);
    }
}

// The frame of the readback was rendered: answers the current requests, then starts the deferred ones
void finishScreenshots(EngineApp& app, const std::shared_ptr<ScreenshotBatch>& batch)
{
    auto requests = std::move(batch->current);
    auto deferred = std::move(batch->deferred);
    batch->current.clear();
    batch->deferred.clear();
    TemplatedImage<ucvec4> image = std::move(batch->image);
    batch->image                 = TemplatedImage<ucvec4>();
    batch->hasImage              = false;
    batch->inFlight              = false;
    if (app.scene())
    {
        app.scene()->setImguiOverride(std::nullopt);
        app.scene()->setLightingDebugOverride(std::nullopt);
    }
    for (auto& r : requests)
    {
        if (r->answered) continue;
        r->answered = true;
        auto answer = std::move(r->answer);
        r->answer   = nullptr;
        answer(image);
    }
    for (auto& r : deferred)
        if (!r->answered) startScreenshot(app, batch, r);
}

void startScreenshot(EngineApp& app, const std::shared_ptr<ScreenshotBatch>& batch,
                     const std::shared_ptr<ScreenshotRequest>& r)
{
    if (r->answered) return;
    if (batch->inFlight)
    {
        // Joins the waiting readback (or adopts it when the deadline answered all its requests)
        if (batch->current.empty() || (batch->gui == r->gui && batch->debug == r->debug))
        {
            batch->gui   = r->gui;
            batch->debug = r->debug;
            batch->current.push_back(r);
        }
        else
            batch->deferred.push_back(r);
        updateBatch(app, *batch);
        return;
    }
    batch->inFlight = true;
    batch->gui      = r->gui;
    batch->debug    = r->debug;
    batch->current.push_back(r);
    updateBatch(app, *batch);
    // The readback (inside renderFrame) only stores the image; the frame callback of the same frame answers. A capture
    // of the app (S20): the view's terrain is completed and the uploads finish first
    app.requestCapture(
        [batch](const TemplatedImage<ucvec4>& img)
        {
            batch->image    = img;
            batch->hasImage = true;
        });
    app.onFrameRendered(
        [&app, batch]()
        {
            if (!batch->hasImage) return false;
            finishScreenshots(app, batch);
            return true;
        });
}

// The deadline answered r: it leaves the batch (its answer and responder are released)
void dropScreenshot(EngineApp& app, ScreenshotBatch& batch, const std::shared_ptr<ScreenshotRequest>& r)
{
    for (auto* list : {&batch.current, &batch.deferred})
        list->erase(std::remove(list->begin(), list->end(), r), list->end());
    r->answer = nullptr;
    r->fail   = nullptr;
    updateBatch(app, batch);
}
}  // namespace

namespace
{
// The terrain block of app_status (S20): the resident tiles, the requests in flight, the upload bytes pending, the CPU
// and GPU memory (MB) and the readiness at 256 m of the view camera
Json terrainStatusJson(EngineApp& app)
{
    const TerrainSystem& t        = app.world().terrain();
    const TerrainSystem::Stats& s = t.stats();
    int resident                  = 0;
    for (const auto& l : s.levels) resident += l.resident;
    double gpu = 0, cpu = double(s.cpuBytes);
    if (app.scene())
    {
        const auto& m = app.scene()->terrainAtlasStats().memory;
        gpu           = double(m.gpuBytes());
        cpu += double(m.cpuBytes());
    }
    const ViewCamera& v = app.viewCamera();
    return Json::object{{"resident", resident},
                        {"pending", s.pending},
                        {"uploads_pending_bytes", double(t.terrainUploadsPending())},
                        {"cpu_mb", cpu / (1 << 20)},
                        {"gpu_mb", gpu / (1 << 20)},
                        {"readiness", t.readiness(v.space, v.position, 256)}};
}
}  // namespace

void registerAppTools(McpServer& server, EngineApp& app)
{
    server.addTool("quit",
                   "Ends the app after the current frame (the exit code is 0). Changes everything: the game stops.",
                   schema({}),
                   [&app](const Json&, McpServer::Respond respond)
                   {
                       app.quit(0);
                       respond(ToolResult::Text(Json(Json::object{{"quitting", true}}).dump()));
                   });

    server.addTool(
        "app_status",
        "Reads the app: headless, rendering enabled, rendered size, frames, frames rendered, fps of the last frame, "
        "steps, the local player (whose input the devices and set_input drive), and the input bindings (actions, held "
        "buttons, axes), the app camera and whether a tool holds it; with --coop also coop {host_step, client_step, "
        "client_peer, delay_steps, dropped_steps} (the tools see the client world), the debug UI's selected tab "
        "(ui_tab; null when the UI is off), terrain {resident, pending, uploads_pending_bytes, cpu_mb, gpu_mb, "
        "readiness (at 256 m of the view camera)} and space_loading (null, or {space, name, progress, frames} while a "
        "loading screen of the view camera's space runs), lighting {time_of_day, ev100, tone_map, debug_view} (null "
        "without a device). Changes nothing.",
        schema({}),
        [&app](const Json&, McpServer::Respond respond)
        {
            Json::object o{{"headless", app.setup().headless},
                           {"rendering", app.scene() != nullptr && app.isRenderingEnabled()},
                           {"frames", double(app.frameCount())},
                           {"frames_rendered", double(app.renderedFrames())},
                           {"step", double(app.world().stepIndex())},
                           {"input", app.input().describe()},
                           {"benchmark", app.options().benchmark},
                           {"camera", cameraJson(app)},
                           {"camera_held", app.cameraHeldByTool()}};
            if (app.scene())
            {
                const ivec2 size = app.scene()->size();
                o["size"]        = Json::array{size.x(), size.y()};
            }
            const Entity p    = app.localPlayer();
            o["local_player"] = p.valid() ? WorldApi::entityJson(p) : Json();
            // --coop (S16b): both worlds' steps (host and client advance interleaved), the delay, the dropped steps
            if (World* host = app.hostWorld())
                o["coop"] = Json::object{{"host_step", double(host->stepIndex())},
                                         {"client_step", double(app.world().stepIndex())},
                                         {"client_peer", int(app.world().localPeer())},
                                         {"delay_steps", app.options().coopDelay},
                                         {"dropped_steps", double(host->droppedSteps())}};
            const Profiler& f = app.frameProfile();
            o["fps"]          = f.counterStats(const_cast<Profiler&>(f).counter("fps"), 1).avg;
            DebugUi& ui       = app.debugUi();
            o["ui_tab"]       = ui.enabled() ? Json(ui.tabName(ui.selected())) : Json();
            o["terrain"]      = terrainStatusJson(app);
            // S36: {time_of_day, ev100, tone_map, debug_view} (null without a device)
            o["lighting"] = lightingStatusJson(app);
            // S21: the view space's load (null: none)
            if (const SpaceLoading* l = app.spaceLoading())
                o["space_loading"] = Json::object{{"space", int(l->space)},
                                                  {"name", app.world().spaceTemplate(l->space).name},
                                                  {"progress", l->progress},
                                                  {"frames", double(l->frames)}};
            else
                o["space_loading"] = Json();
            respond(ToolResult::Text(Json(o).dump()));
        });

    server.addTool(
        "set_ui_tab",
        "Selects a tab of the developer UI, as its F-key does: game (F1, the game without a panel), performance (F2), "
        "engine (F3), console (F4, the log with the MCP call log), lighting (F5: time, weather, sun and moon, "
        "exposure) "
        "or a game tab by name (case-insensitive). Without "
        "tab it only reads. Returns the selected tab, the tabs with their keys and whether the UI is visible (G hides "
        "all ImGui; this tool does not show it again). Use it before screenshot(gui: true) to capture a tab. Changes "
        "only the UI, not the world.",
        schema({{"tab", McpServer::StringProperty("The tab's name")}}),
        [&app](const Json& a, McpServer::Respond respond)
        {
            if (unknownArgument(a, {"tab"}, respond)) return;
            DebugUi& ui = app.debugUi();
            if (!ui.enabled()) return respond(ToolResult::Error("The debug UI is off (EngineAppSetup::debugUi)"));
            Json::array tabs;
            std::string names;
            for (int i = 0; i < ui.tabCount(); ++i)
            {
                Json::object t{{"name", ui.tabName(i)}};
                if (ui.tabKey(i) > 0) t["key"] = "F" + std::to_string(ui.tabKey(i));
                tabs.push_back(t);
                names += (names.empty() ? "" : ", ") + ui.tabName(i);
            }
            if (!a["tab"].is_null())
            {
                if (!a["tab"].is_string()) return respond(ToolResult::Error("tab: a string"));
                const int t = ui.findTab(a["tab"].string_value());
                if (t < 0)
                    return respond(
                        ToolResult::Error("Unknown tab \"" + a["tab"].string_value() + "\" (tabs: " + names + ")"));
                ui.select(t);
            }
            const bool visible = app.scene() && app.scene()->imguiVisible() && app.setup().render.imgui;
            respond(ToolResult::Text(
                Json(Json::object{{"tab", ui.tabName(ui.selected())}, {"tabs", tabs}, {"visible", visible}}).dump()));
        });

    server.addTool(
        "set_input",
        "Drives the local player like a player at the keyboard: axes {name: value} (e.g. move_x, move_z in -1..1), "
        "held button names and pressed action names (queued once), for the next steps world steps (default 1; 0: until "
        "set_input is called again; a paused world keeps the input until steps run). The names are the input "
        "bindings of app_status. Changes the local player's PlayerInput and queues its actions.",
        schema({{"axes", Json::object{{"type", "object"}, {"description", "{axis name: value}"}}},
                {"held", Json::object{{"type", "array"}, {"items", Json::object{{"type", "string"}}}}},
                {"pressed", Json::object{{"type", "array"}, {"items", Json::object{{"type", "string"}}}}},
                {"steps",
                 Json::object{{"type", "integer"}, {"description", "World steps (default 1; 0: until changed)"}}}}),
        [&app](const Json& args, McpServer::Respond respond)
        {
            for (auto& kv : args.object_items())
                if (kv.first != "axes" && kv.first != "held" && kv.first != "pressed" && kv.first != "steps")
                    return respond(ToolResult::Error("unknown argument \"" + kv.first + "\""));
            if (!app.localPlayer().valid())
                return respond(ToolResult::Error("set_input: the app has no local player (app_status)"));
            VirtualInput v;
            if (!args["axes"].is_null())
            {
                if (!args["axes"].is_object()) return respond(ToolResult::Error("axes: expected {name: value}"));
                for (auto& kv : args["axes"].object_items())
                {
                    if (!app.input().hasAxis(kv.first))
                        return respond(ToolResult::Error("axes: no axis \"" + kv.first + "\" (app_status)"));
                    if (!kv.second.is_number() || !std::isfinite(kv.second.number_value()))
                        return respond(ToolResult::Error("axes." + kv.first + ": expected a number"));
                    v.axes[kv.first] = kv.second.number_value();
                }
            }
            for (const char* key : {"held", "pressed"})
            {
                if (args[key].is_null()) continue;
                if (!args[key].is_array()) return respond(ToolResult::Error(std::string(key) + ": expected names"));
                for (const Json& n : args[key].array_items())
                {
                    if (!n.is_string()) return respond(ToolResult::Error(std::string(key) + ": expected names"));
                    const ActionList& actions = app.world().actionList();
                    if (!actions.find(n.string_value()) && !actions.findHeld(n.string_value()))
                        return respond(ToolResult::Error(std::string(key) + ": no action or held button \"" +
                                                         n.string_value() + "\""));
                    (std::string(key) == "held" ? v.held : v.pressed).push_back(n.string_value());
                }
            }
            int steps = 1;
            if (!args["steps"].is_null())
            {
                const double f = args["steps"].number_value();
                if (!args["steps"].is_number() || f != std::floor(f) || f < 0 || f > 1e6)
                    return respond(ToolResult::Error("steps: an integer 0..1000000"));
                steps = int(f);
            }
            app.setVirtualInput(v, steps);
            respond(ToolResult::Text(Json(Json::object{{"steps", steps}}).dump()));
        });

    auto batch = std::make_shared<ScreenshotBatch>();
    // S20: the app bounds a frame's terrain completion by the earliest deadline of the waiting requests, and answers
    // the expired ones (marked answered and dropped from the batch before the poll runs again: the deadline listener
    // does not answer them a second time)
    app.setScreenshotDeadlineHooks(
        [batch]() -> std::optional<std::chrono::steady_clock::time_point>
        {
            std::optional<std::chrono::steady_clock::time_point> earliest;
            for (const auto* list : {&batch->current, &batch->deferred})
                for (const auto& r : *list)
                    if (!r->answered && (!earliest || r->deadline < *earliest)) earliest = r->deadline;
            return earliest;
        },
        [&app, batch](std::chrono::steady_clock::time_point limit)
        {
            std::vector<std::shared_ptr<ScreenshotRequest>> expired;
            for (const auto* list : {&batch->current, &batch->deferred})
                for (const auto& r : *list)
                    if (!r->answered && r->deadline <= limit) expired.push_back(r);
            for (const auto& r : expired)
            {
                r->answered = true;
                auto fail   = std::move(r->fail);
                dropScreenshot(app, *batch, r);
                if (fail) fail("screenshot: terrain not complete before the deadline");
            }
        });
    server.addTool(
        "screenshot",
        "Renders the next frame and returns it as an image (jpg default, or png), downscaled to max_width when wider "
        "(box filter; default 1280, 64..8192; never upscaled), followed by the camera of that frame, the step and the "
        "frame "
        "number. The GUI is hidden unless gui=true (only in that frame: the app's GUI visibility is not changed). An "
        "error without a device "
        "or when no frame is rendered before the deadline (rendering disabled, a minimized window). The frame shows "
        "the "
        "complete terrain of the view (streamed first; an error if that does not finish before the deadline). S36: "
        "lighting_view (a set_lighting_debug view name) and overlays (a list of overlay names) render that frame with "
        "the debug view and overlays (a one-frame override; the next frame is back to the user's). Changes nothing.",
        schema(
            {{"gui", McpServer::BooleanProperty("Include the ImGui windows (default false)")},
             {"format", McpServer::StringProperty("Image format (default jpg)", {"jpg", "png"})},
             {"max_width", McpServer::IntegerProperty("Downscale to at most this width (default 1280, 64..8192)")},
             {"lighting_view", McpServer::StringProperty("The lighting debug view of this frame (set_lighting_debug)")},
             {"overlays", Json::object{{"type", "array"},
                                       {"description", "The lighting overlays of this frame"},
                                       {"items", Json::object{{"type", "string"}}}}}}),
        [&app, &server, batch](const Json& args, McpServer::Respond respond)
        {
            if (unknownArgument(args, {"gui", "format", "max_width", "lighting_view", "overlays"}, respond)) return;
            if (!app.scene()) return respond(ToolResult::Error("screenshot: this app renders no frames (no device)"));
            if (!args["gui"].is_null() && !args["gui"].is_bool()) return respond(ToolResult::Error("gui: a bool"));
            const bool gui           = args["gui"].bool_value();
            const std::string format = args["format"].is_null() ? "jpg" : args["format"].string_value();
            if (format != "jpg" && format != "png") return respond(ToolResult::Error("format: jpg or png"));
            int maxWidth = 1280;
            if (!args["max_width"].is_null())
            {
                const double w = args["max_width"].number_value();
                if (!args["max_width"].is_number() || w != std::floor(w) || w < 64 || w > 8192)
                    return respond(ToolResult::Error("max_width: an integer 64..8192"));
                maxWidth = int(w);
            }
            auto r = std::make_shared<ScreenshotRequest>();
            if (!args["lighting_view"].is_null() || !args["overlays"].is_null())
            {
                Vulkan::LightingDebugState debug;
                std::string error;
                if (!parseLightingDebug(args["lighting_view"], args["overlays"], debug, error))
                    return respond(ToolResult::Error(error));
                r->debug = debug;
            }
            const auto deadline = answerDeadline(server);
            r->gui              = gui;
            r->deadline         = deadline;
            r->fail             = [respond](const std::string& message) { respond(ToolResult::Error(message)); };
            r->answer           = [&app, respond, format, maxWidth](const TemplatedImage<ucvec4>& image)
            {
                if (image.width == 0) return respond(ToolResult::Error("screenshot: the frame could not be read back"));
                auto small = downscale(image, maxWidth);
                small.set_compression_quality(85);
                const auto data = small.saveToMemory(format);
                if (data.empty()) return respond(ToolResult::Error("screenshot: could not encode as " + format));
                ToolResult result = ToolResult::Image(data, format == "png" ? "image/png" : "image/jpeg");
                result.addText(Json(Json::object{{"camera", cameraJson(app)},
                                                 {"step", double(app.world().stepIndex())},
                                                 {"frame", double(app.renderedFrames())}})
                                   .dump());
                respond(result);
            };
            // The deadline answers once if no frame came (the request leaves the batch, the GUI override is cleared)
            server.addPollListener(
                [&app, batch, r, respond, deadline]()
                {
                    if (r->answered) return false;
                    if (std::chrono::steady_clock::now() < deadline) return true;
                    r->answered = true;
                    dropScreenshot(app, *batch, r);
                    respond(
                        ToolResult::Error("screenshot: no frame was rendered before the deadline (rendering "
                                          "disabled or the window minimized)"));
                    return false;
                });
            startScreenshot(app, batch, r);
        });

    server.addTool(
        "set_camera",
        "Places the app camera: position [x, y, z], the view from look_at [x, y, z] (not the position) or from yaw_deg "
        "(0 looks along -z, 90 along +x; within +-3600) and pitch_deg (-90..90, positive looks up); missing values "
        "stay. Takes the camera from the game (camera "
        "hold: games that drive the camera stop doing so) until release: true (alone). While held, relative moves of "
        "the input (keys, set_input) use this camera's heading. The field of view is a game setting. Returns the "
        "camera. space: the space the camera views (default: the current one; describe_world lists them). Changes the "
        "camera.",
        schema({{"space", McpServer::IntegerProperty("The space of the camera (default: the current one)")},
                {"position", McpServer::Vec3Property("Camera position [x, y, z] in meters")},
                {"look_at", McpServer::Vec3Property("Point to look at [x, y, z]")},
                {"yaw_deg", McpServer::NumberProperty("Heading in degrees, 0 = -z, 90 = +x")},
                {"pitch_deg", McpServer::NumberProperty("Pitch in degrees, positive looks up")},
                {"release", McpServer::BooleanProperty("Give the camera back to the game (no other argument)")}}),
        [&app](const Json& args, McpServer::Respond respond)
        {
            if (unknownArgument(args, {"space", "position", "look_at", "yaw_deg", "pitch_deg", "release"}, respond))
                return;
            if (!args["release"].is_null())
            {
                if (!args["release"].is_bool() || !args["release"].bool_value() || args.object_items().size() != 1)
                    return respond(ToolResult::Error("release: true, without other arguments"));
                app.releaseCamera();
                return respond(ToolResult::Text(cameraJson(app).dump()));
            }
            Vec3 pos    = app.viewCamera().position;
            Vec3 dir    = cameraForward(app.viewCamera());
            Vec3 target = Vec3::Zero();
            bool hasPos, hasTarget, hasYaw, hasPitch;
            double yaw = 0, pitch = 0;
            if (!vec3Argument(args, "position", pos, hasPos, respond) ||
                !vec3Argument(args, "look_at", target, hasTarget, respond) ||
                !numberArgument(args, "yaw_deg", 3600, yaw, hasYaw, respond) ||
                !numberArgument(args, "pitch_deg", 90, pitch, hasPitch, respond))
                return;
            if (hasTarget && (target - pos).norm() < 1e-4)
                return respond(ToolResult::Error("look_at must differ from the position"));
            // S21: the space of the app's view world (the client world with --coop)
            SpaceId space = app.viewCamera().space;
            if (!args["space"].is_null())
            {
                const double s     = args["space"].number_value();
                const size_t count = app.world().spaceCount();
                if (!args["space"].is_number() || s != std::floor(s) || s < 0 || s >= double(count))
                    return respond(ToolResult::Error("space: an integer in 0.." + std::to_string(count - 1) + " (" +
                                                     std::to_string(count) + " spaces)"));
                space = SpaceId(s);
            }
            if (hasTarget)
                dir = target - pos;
            else if (hasYaw || hasPitch)
            {
                double y0, p0;
                yawPitchFromDirection(dir, y0, p0);
                dir = directionFromYawPitch(hasYaw ? yaw : y0, hasPitch ? pitch : p0);
            }
            lookFrom(app, pos, dir, space);
            respond(ToolResult::Text(cameraJson(app).dump()));
        });

    server.addTool(
        "move_camera",
        "Moves the app camera relative to the view: forward and right in meters along the ground plane, up in meters, "
        "then turns by yaw_deg (positive: right) and pitch_deg (positive: up; the result is clamped to +-89). Takes "
        "the "
        "camera from the game like set_camera. Returns the camera. Changes the camera.",
        schema({{"forward", McpServer::NumberProperty("Meters forward (horizontal)")},
                {"right", McpServer::NumberProperty("Meters to the right")},
                {"up", McpServer::NumberProperty("Meters up")},
                {"yaw_deg", McpServer::NumberProperty("Turn right by this angle")},
                {"pitch_deg", McpServer::NumberProperty("Turn up by this angle")}}),
        [&app](const Json& args, McpServer::Respond respond)
        {
            if (unknownArgument(args, {"forward", "right", "up", "yaw_deg", "pitch_deg"}, respond)) return;
            double forward = 0, right = 0, up = 0, dyaw = 0, dpitch = 0;
            bool g;
            if (!numberArgument(args, "forward", kCameraLimit, forward, g, respond) ||
                !numberArgument(args, "right", kCameraLimit, right, g, respond) ||
                !numberArgument(args, "up", kCameraLimit, up, g, respond) ||
                !numberArgument(args, "yaw_deg", 3600, dyaw, g, respond) ||
                !numberArgument(args, "pitch_deg", 3600, dpitch, g, respond))
                return;
            Vec3 pos = app.viewCamera().position;
            double yaw, pitch;
            yawPitchFromDirection(cameraForward(app.viewCamera()), yaw, pitch);
            const Vec3 flatForward = directionFromYawPitch(yaw, 0);
            const Vec3 rightDir    = directionFromYawPitch(yaw + 90, 0);
            pos += flatForward * forward + rightDir * right + Vec3(0, 1, 0) * up;
            for (int i = 0; i < 3; ++i) pos(i) = std::min(std::max(pos(i), -kCameraLimit), kCameraLimit);
            yaw += dyaw;
            pitch = std::min(std::max(pitch + dpitch, -89.0), 89.0);
            lookFrom(app, pos, directionFromYawPitch(yaw, pitch));
            respond(ToolResult::Text(cameraJson(app).dump()));
        });

    server.addTool(
        "list_scenarios",
        "The game's scenarios (registration order) with local_only, optionally those whose name contains filter. "
        "Changes nothing.",
        schema({{"filter", McpServer::StringProperty("Part of the name")}}),
        [&app](const Json& args, McpServer::Respond respond)
        {
            if (unknownArgument(args, {"filter"}, respond)) return;
            if (!args["filter"].is_null() && !args["filter"].is_string())
                return respond(ToolResult::Error("filter: a string"));
            ScenarioList list;
            if (app.setup().scenarios) app.setup().scenarios(list);
            Json::array out;
            for (const ScenarioList::Entry* e : list.match(args["filter"].string_value()))
                out.push_back(Json::object{{"name", e->name}, {"local_only", e->localOnly}});
            respond(ToolResult::Text(Json(Json::object{{"scenarios", out}}).dump()));
        });

    server.addTool(
        "run_scenario",
        "Runs one of the game's scenarios headless in this call (frames stop meanwhile; scenario log lines go to logs) "
        "in the given variants (default [\"Plain\"]; Repeat, SaveLoad) and returns per variant passed, failures, "
        "steps, seconds and, for a failed one, its report. The scenario uses the live world's data (run reload_data "
        "first after editing data files). keep: true stores the Plain run's end state as the snapshot "
        "scenario_<name>; load: true also loads it into the live world, paused, after saving the live world as "
        "before_scenario (kept while the live world comes from run_scenario loads; a manual load_snapshot counts as "
        "another load); load_snapshot before_scenario goes back. At the deadline the running variant stops (failure "
        "\"aborted: stopped: tool time budget\"; checked after each step: a scenario that does not step blocks until "
        "it returns, and its end state is then not stored). The host + client variants (HostClient, HostClientDelay, "
        "HostClientAsync) run a host and a client over loopback (not for local-only scenarios). With --coop keep and "
        "load are refused (the live world is a client). An assertion in scenario code ends the app. Changes the "
        "snapshots and, with load, the live world.",
        schema({{"name", McpServer::StringProperty("The scenario (list_scenarios)")},
                {"variants",
                 Json::object{{"type", "array"},
                              {"items", McpServer::StringProperty("", {"Plain", "Repeat", "SaveLoad", "HostClient",
                                                                       "HostClientDelay", "HostClientAsync"})}}},
                {"keep", McpServer::BooleanProperty("Store the end state as the snapshot scenario_<name>")},
                {"load", McpServer::BooleanProperty("Store and load the end state into the live world")}}),
        [&app, &server](const Json& args, McpServer::Respond respond)
        {
            if (unknownArgument(args, {"name", "variants", "keep", "load"}, respond)) return;
            if (!app.setup().scenarios) return respond(ToolResult::Error("run_scenario: the game has no scenarios"));
            if (!args["name"].is_string()) return respond(ToolResult::Error("name: a scenario name"));
            for (const char* key : {"keep", "load"})
                if (!args[key].is_null() && !args[key].is_bool())
                    return respond(ToolResult::Error(std::string(key) + ": a bool"));
            const bool keep = args["keep"].bool_value(), load = args["load"].bool_value();
            ScenarioList list;
            app.setup().scenarios(list);
            const ScenarioList::Entry* entry = list.find(args["name"].string_value());
            if (!entry)
                return respond(ToolResult::Error("run_scenario: no scenario \"" + args["name"].string_value() +
                                                 "\" (list_scenarios)"));
            std::vector<ScenarioVariant> variants;
            if (args["variants"].is_null())
                variants.push_back(ScenarioVariant::Plain);
            else
            {
                const char* names = "Plain, Repeat, SaveLoad, HostClient, HostClientDelay, HostClientAsync";
                if (!args["variants"].is_array() || args["variants"].array_items().empty())
                    return respond(ToolResult::Error(std::string("variants: a non-empty list of ") + names));
                for (const Json& v : args["variants"].array_items())
                {
                    bool found = false;
                    for (ScenarioVariant x : {ScenarioVariant::Plain, ScenarioVariant::Repeat,
                                              ScenarioVariant::SaveLoad, ScenarioVariant::HostClient,
                                              ScenarioVariant::HostClientDelay, ScenarioVariant::HostClientAsync})
                        if (v.is_string() && v.string_value() == variantName(x))
                        {
                            // A local-only scenario has no host + client variants
                            if (entry->localOnly && networkOf(x).enabled)
                                return respond(ToolResult::Error("variants: " + v.string_value() + ": \"" +
                                                                 entry->name + "\" is local-only"));
                            if (std::find(variants.begin(), variants.end(), x) == variants.end()) variants.push_back(x);
                            found = true;
                        }
                    if (!found)
                        return respond(ToolResult::Error("variants: unknown variant " + v.dump() + " (" + names + ")"));
                }
            }
            const bool plain = std::find(variants.begin(), variants.end(), ScenarioVariant::Plain) != variants.end();
            if ((keep || load) && !plain)
                return respond(ToolResult::Error("run_scenario: keep and load need the Plain variant"));
            if ((keep || load) && app.coop())
                return respond(ToolResult::Error(
                    "run_scenario: keep and load are refused in --coop (the live world is a client; it cannot load)"));
            // The world the scenario's data come from: with --coop the hidden host (its data store, its steps)
            World& live = app.startWorld();
            if (!live.canSave())
                return respond(ToolResult::Error("run_scenario: only between steps, not inside a game hook"));
            WorldApi& api              = app.api();
            const std::string snapName = "scenario_" + entry->name;
            if ((keep || load) && snapName.size() > 128)
                return respond(ToolResult::Error("run_scenario: the scenario name is too long for a snapshot name"));
            const bool saveBefore = load && !(app.liveWorldFromScenario() && api.hasSnapshot("before_scenario"));
            // The snapshot slots before the run (the bytes of the scenario's state are known only after it)
            size_t newNames = 0;
            if ((keep || load) && !api.hasSnapshot(snapName)) newNames++;
            if (saveBefore && !api.hasSnapshot("before_scenario")) newNames++;
            if (api.snapshotCount() + newNames > WorldApi::kMaxSnapshotCount)
                return respond(ToolResult::Error("run_scenario: the snapshots are full (delete_snapshot frees one)"));
            if (saveBefore)
            {
                // Before the run: the live world cannot change during this call, and its bytes are checked now
                const ApiResult saved =
                    api.call("save_snapshot", Json::object{{"name", "before_scenario"}, {"overwrite", true}});
                if (!saved.ok) return respond(ToolResult::Error("run_scenario: " + saved.error));
            }

            app.countScenarioRun();
            ScenarioOptions options;
            options.deadline         = answerDeadline(server);
            WorldSetup setup         = app.setup().world;
            setup.sharedData         = live.sharedDataStore();
            setup.sharedDataReadOnly = true;
            Json::array results;
            std::vector<uint8_t> finalSave;
            uint64_t finalStep = 0, finalHash = 0;
            bool plainLate = false;  // the Plain run finished after the deadline (the client may have given up)
            for (ScenarioVariant v : variants)
            {
                if (std::chrono::steady_clock::now() >= *options.deadline)
                {
                    results.push_back(Json::object{{"variant", variantName(v)}, {"skipped", true}});
                    continue;
                }
                ScenarioResult r = runScenario(setup, *entry, v, options);
                Json::array failures;
                for (const std::string& f : r.failures) failures.push_back(f);
                Json::object o{{"variant", variantName(v)},
                               {"passed", r.passed},
                               {"failures", failures},
                               {"steps", double(r.steps)},
                               {"seconds", r.seconds}};
                if (!r.passed)
                {
                    // Cut at a UTF-8 character boundary (no continuation byte first)
                    size_t cut = std::min<size_t>(r.report.size(), 16384);
                    while (cut > 0 && cut < r.report.size() && (uint8_t(r.report[cut]) & 0xC0) == 0x80) cut--;
                    o["report"] = cut < r.report.size() ? r.report.substr(0, cut) + "..." : r.report;
                }
                results.push_back(o);
                // A run that finished after the deadline (no step after it saw the clock, e.g. a long final check):
                // its end state is neither stored nor loaded (the client may already have its timeout)
                if (v == ScenarioVariant::Plain && !r.finalSave.empty() &&
                    std::chrono::steady_clock::now() >= *options.deadline)
                    plainLate = true;
                else if (v == ScenarioVariant::Plain)
                {
                    finalSave = std::move(r.finalSave);
                    finalStep = r.finalStep;
                    finalHash = r.finalHash;
                }
            }
            Json::object out{{"scenario", entry->name}, {"results", results}};
            if ((keep || load) && !finalSave.empty())
            {
                const ApiResult stored = api.putSnapshot(snapName, std::move(finalSave), finalStep, finalHash, true);
                if (!stored.ok)
                    out["snapshot_error"] = stored.error;
                else
                {
                    out["snapshot"] = snapName;
                    char hex[32];
                    std::snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)finalHash);
                    out["snapshot_hash"] = std::string(hex);
                    if (load)
                    {
                        app.setScenarioLoadInProgress(true);
                        const ApiResult loaded =
                            api.call("load_snapshot", Json::object{{"name", snapName}, {"paused", true}});
                        app.setScenarioLoadInProgress(false);
                        if (!loaded.ok)
                            out["load_error"] = loaded.error;
                        else
                        {
                            Json::object l = loaded.value.object_items();
                            char hex[32];
                            std::snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)app.world().hash());
                            l["hash"]     = std::string(hex);
                            out["loaded"] = l;
                        }
                    }
                }
            }
            else if ((keep || load) && plain)
                out["snapshot_error"] = plainLate
                                            ? "the Plain run finished after the tool's deadline (its end state is "
                                              "not stored)"
                                            : "the Plain run did not finish (no end state)";
            app.restartFrameClock();
            respond(ToolResult::Text(exactToolJson(Json(out))));
        });
}

}  // namespace Engine
}  // namespace Saiga
