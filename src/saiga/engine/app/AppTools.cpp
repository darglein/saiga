/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The app's MCP tools: quit, app_status, set_input (S13).

#include "saiga/engine/app/AppTools.h"

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include <cmath>

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
        "buttons, axes). Changes nothing.",
        schema({}),
        [&app](const Json&, McpServer::Respond respond)
        {
            Json::object o{{"headless", app.setup().headless},
                           {"rendering", app.scene() != nullptr && app.isRenderingEnabled()},
                           {"frames", double(app.frameCount())},
                           {"frames_rendered", double(app.renderedFrames())},
                           {"step", double(app.world().stepIndex())},
                           {"input", app.input().describe()},
                           {"benchmark", app.options().benchmark}};
            if (app.scene())
            {
                const ivec2 size = app.scene()->size();
                o["size"]        = Json::array{size.x(), size.y()};
            }
            const Entity p    = app.localPlayer();
            o["local_player"] = p.valid() ? WorldApi::entityJson(p) : Json();
            const Profiler& f = app.frameProfile();
            o["fps"]          = f.counterStats(const_cast<Profiler&>(f).counter("fps"), 1).avg;
            respond(ToolResult::Text(Json(o).dump()));
        });

    server.addTool(
        "set_input",
        "Drives the local player like a player at the keyboard: axes {name: value} (e.g. move_x, move_z in -1..1), "
        "held "
        "button names and pressed action names (once), for the next frames frames (default 1; 0: until set_input is "
        "called again). The names are the input bindings of app_status. Changes the local player's PlayerInput and "
        "queues its actions.",
        schema(
            {{"axes", Json::object{{"type", "object"}, {"description", "{axis name: value}"}}},
             {"held", Json::object{{"type", "array"}, {"items", Json::object{{"type", "string"}}}}},
             {"pressed", Json::object{{"type", "array"}, {"items", Json::object{{"type", "string"}}}}},
             {"frames", Json::object{{"type", "integer"}, {"description", "Frames (default 1; 0: until changed)"}}}}),
        [&app](const Json& args, McpServer::Respond respond)
        {
            for (auto& kv : args.object_items())
                if (kv.first != "axes" && kv.first != "held" && kv.first != "pressed" && kv.first != "frames")
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
            int frames = 1;
            if (!args["frames"].is_null())
            {
                const double f = args["frames"].number_value();
                if (!args["frames"].is_number() || f != std::floor(f) || f < 0 || f > 1e6)
                    return respond(ToolResult::Error("frames: an integer 0..1000000"));
                frames = int(f);
            }
            app.setVirtualInput(v, frames);
            respond(ToolResult::Text(Json(Json::object{{"frames", frames}}).dump()));
        });
}

}  // namespace Engine
}  // namespace Saiga
