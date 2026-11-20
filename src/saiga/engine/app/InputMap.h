/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/core/util/json11.hpp"
#include "saiga/engine/gameplay/ActionList.h"
#include "saiga/export.h"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// The state of the input devices in one frame (keys by GLFW key code)
struct InputState
{
    std::array<bool, 512> keys{};
    std::array<bool, 8> mouseButtons{};
    Vec2 mouseDelta = Vec2::Zero();  // pixels since the last frame
};

// A virtual input device (the set_input tool): axis values, held buttons and actions pressed in this frame
struct VirtualInput
{
    std::map<std::string, double> axes;
    std::vector<std::string> held, pressed;
};

/**
 * Devices to named actions, held buttons and axes (docs/engine_design.md §8.3; S13: keyboard and mouse, gamepads in
 * S35). The names come from the game's ActionList; the bindings from data/<game>/input.json:
 *
 *   {"bindings": {"jump": ["Space"], "walk": ["LeftShift"]},
 *    "axes": {"move_x": {"negative": ["A"], "positive": ["D"]}, "look_x": {"mouse": "x", "scale": 0.002}},
 *    "move": ["move_x", "move_z"]}
 *
 * bindings: an action or held button name -> keys (GLFW key names: A..Z, 0..9, Space, LeftShift, Up, F1, ...) or mouse
 * buttons (Mouse1..Mouse8); axes: a name -> keys for -1 and +1, or a mouse axis ("x", "y") times scale; move: the two
 * axes of PlayerInput.move (right, forward). Unknown keys, names and fields are errors naming the file and the key.
 */
class SAIGA_ENGINE_API InputMap
{
   public:
    InputMap() = default;
    // Reads input.json; user (optional, may be missing) overrides bindings and axes by name. False and an error naming
    // the file and the key if something is wrong.
    bool load(const std::string& file, const std::string& user, const ActionList& actions, std::string& error);
    // The game's actions and held buttons (also without bindings: virtual input uses the names)
    void setActions(const ActionList& actions) { actionList = &actions; }
    // From JSON (tests); file names the source in errors
    bool parse(const json11::Json& json, const std::string& file, const ActionList& actions, std::string& error);

    // A new frame of input; keyboardCaptured (ImGui): device keys and buttons count as released
    void update(const InputState& state, bool keyboardCaptured, const VirtualInput& virtualInput = {});

    bool held(const std::string& name) const;
    // Held now, not in the previous frame
    bool pressed(const std::string& name) const;
    double axis(const std::string& name) const;
    // The move axes (x: right, y: forward), length at most 1
    Vec2 move() const;
    // Bits of the held buttons of the ActionList that are held
    uint32_t heldBits() const;
    // The action IDs pressed in this frame, in ActionList order
    std::vector<uint32_t> pressedActions() const;
    // Nothing is held, no axis is off zero
    bool neutral() const;

    // The bindings as JSON (app_status)
    json11::Json describe() const;
    // GLFW key code of a key name, -1 if unknown
    static int keyCode(const std::string& name);

   private:
    struct Axis
    {
        std::vector<int> negative, positive;
        int mouseAxis = -1;  // 0: x, 1: y
        double scale  = 1;
    };
    std::map<std::string, std::vector<int>> bindings;  // name -> key codes (mouse buttons: 1000 + button)
    std::map<std::string, Axis> axes;
    std::string moveX, moveZ;
    const ActionList* actionList = nullptr;
    std::map<std::string, bool> heldNow, heldBefore;
    std::map<std::string, double> axisValues;
};

}  // namespace Engine
}  // namespace Saiga
