/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// InputMap: keys and mouse to the game's named actions, held buttons and axes (S13).

#include "saiga/engine/app/InputMap.h"

#include "saiga/engine/data/DataFile.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace Saiga
{
namespace Engine
{
namespace
{
using Json = json11::Json;

// GLFW key codes (the values of glfw3.h; no GLFW include in the engine)
const std::map<std::string, int>& keyTable()
{
    static const std::map<std::string, int> table = []
    {
        std::map<std::string, int> t;
        for (char c = 'A'; c <= 'Z'; ++c) t[std::string(1, c)] = int(c);
        for (char c = '0'; c <= '9'; ++c) t[std::string(1, c)] = int(c);
        t["Space"]        = 32;
        t["Escape"]       = 256;
        t["Enter"]        = 257;
        t["Tab"]          = 258;
        t["Backspace"]    = 259;
        t["Right"]        = 262;
        t["Left"]         = 263;
        t["Down"]         = 264;
        t["Up"]           = 265;
        t["LeftShift"]    = 340;
        t["LeftControl"]  = 341;
        t["LeftAlt"]      = 342;
        t["RightShift"]   = 344;
        t["RightControl"] = 345;
        t["RightAlt"]     = 346;
        for (int i = 1; i <= 12; ++i) t["F" + std::to_string(i)] = 289 + i;
        for (int i = 1; i <= 8; ++i) t["Mouse" + std::to_string(i)] = 1000 + i - 1;
        return t;
    }();
    return table;
}

bool keyList(const Json& j, const std::string& where, std::vector<int>& out, std::string& error)
{
    if (!j.is_array())
    {
        error = where + ": expected an array of key names";
        return false;
    }
    for (const Json& k : j.array_items())
    {
        const int code = k.is_string() ? InputMap::keyCode(k.string_value()) : -1;
        if (code < 0)
        {
            error = where + ": unknown key " + k.dump();
            return false;
        }
        out.push_back(code);
    }
    return true;
}

bool down(const InputState& s, int code)
{
    if (code >= 1000) return code - 1000 < int(s.mouseButtons.size()) && s.mouseButtons[size_t(code - 1000)];
    return code >= 0 && code < int(s.keys.size()) && s.keys[size_t(code)];
}
}  // namespace

int InputMap::keyCode(const std::string& name)
{
    auto it = keyTable().find(name);
    return it == keyTable().end() ? -1 : it->second;
}

bool InputMap::load(const std::string& file, const std::string& user, const ActionList& actions, std::string& error)
{
    Json json;
    if (!readJsonFile(file, json, error) || !parse(json, file, actions, error)) return false;
    if (user.empty() || !std::ifstream(user).good()) return true;
    // The user copy overrides the bindings and axes it names
    Json userJson;
    if (!readJsonFile(user, userJson, error)) return false;
    InputMap merged = *this;
    if (!merged.parse(userJson, user, actions, error)) return false;
    for (auto& kv : merged.bindings) bindings[kv.first] = kv.second;
    for (auto& kv : merged.axes) axes[kv.first] = kv.second;
    if (userJson["move"].is_array())
    {
        moveX = merged.moveX;
        moveZ = merged.moveZ;
    }
    return true;
}

bool InputMap::parse(const Json& json, const std::string& file, const ActionList& actions, std::string& error)
{
    if (!json.is_object())
    {
        error = file + ": expected an object";
        return false;
    }
    actionList = &actions;
    std::map<std::string, std::vector<int>> b;
    std::map<std::string, Axis> a;
    std::string mx = moveX, mz = moveZ;
    for (const auto& kv : json.object_items())
    {
        const std::string where = file + ": " + kv.first;
        if (kv.first == "bindings")
        {
            if (!kv.second.is_object())
            {
                error = where + ": expected an object of action names";
                return false;
            }
            for (const auto& binding : kv.second.object_items())
            {
                if (!actions.find(binding.first) && !actions.findHeld(binding.first))
                {
                    error = where + "." + binding.first + ": not a registered action or held button";
                    return false;
                }
                if (!keyList(binding.second, where + "." + binding.first, b[binding.first], error)) return false;
            }
        }
        else if (kv.first == "axes")
        {
            if (!kv.second.is_object())
            {
                error = where + ": expected an object of axis names";
                return false;
            }
            for (const auto& axis : kv.second.object_items())
            {
                const std::string aw = where + "." + axis.first;
                if (!axis.second.is_object())
                {
                    error = aw + ": expected an object";
                    return false;
                }
                Axis x;
                for (const auto& f : axis.second.object_items())
                {
                    if (f.first == "negative" || f.first == "positive")
                    {
                        if (!keyList(f.second, aw + "." + f.first, f.first == "negative" ? x.negative : x.positive,
                                     error))
                            return false;
                    }
                    else if (f.first == "mouse")
                    {
                        if (f.second != Json("x") && f.second != Json("y"))
                        {
                            error = aw + ".mouse: \"x\" or \"y\"";
                            return false;
                        }
                        x.mouseAxis = f.second == Json("x") ? 0 : 1;
                    }
                    else if (f.first == "scale")
                    {
                        if (!f.second.is_number() || !std::isfinite(f.second.number_value()))
                        {
                            error = aw + ".scale: expected a number";
                            return false;
                        }
                        x.scale = f.second.number_value();
                    }
                    else
                    {
                        error = aw + "." + f.first + ": unknown field (fields: negative, positive, mouse, scale)";
                        return false;
                    }
                }
                a[axis.first] = x;
            }
        }
        else if (kv.first == "move")
        {
            if (!kv.second.is_array() || kv.second.array_items().size() != 2 || !kv.second[0].is_string() ||
                !kv.second[1].is_string())
            {
                error = where + ": expected [right axis, forward axis]";
                return false;
            }
            mx = kv.second[0].string_value();
            mz = kv.second[1].string_value();
        }
        else
        {
            error = where + ": unknown field (fields: bindings, axes, move)";
            return false;
        }
    }
    for (const std::string& m : {mx, mz})
        if (!m.empty() && !a.count(m) && !axes.count(m))
        {
            error = file + ": move: the axis " + m + " is not defined";
            return false;
        }
    bindings = b;
    axes     = a;
    moveX    = mx;
    moveZ    = mz;
    return true;
}

void InputMap::update(const InputState& state, bool keyboardCaptured, const VirtualInput& v)
{
    heldBefore = heldNow;
    heldNow.clear();
    for (const auto& kv : bindings)
    {
        bool h = false;
        if (!keyboardCaptured)
            for (int code : kv.second) h = h || down(state, code);
        heldNow[kv.first] = h;
    }
    for (const std::string& name : v.held) heldNow[name] = true;
    axisValues.clear();
    for (const auto& kv : axes)
    {
        double value = 0;
        if (!keyboardCaptured)
        {
            for (int code : kv.second.negative) value -= down(state, code) ? 1 : 0;
            for (int code : kv.second.positive) value += down(state, code) ? 1 : 0;
            if (kv.second.mouseAxis >= 0) value += state.mouseDelta[kv.second.mouseAxis] * kv.second.scale;
        }
        if (kv.second.mouseAxis < 0) value = std::max(-1.0, std::min(1.0, value));
        axisValues[kv.first] = value;
    }
    for (const auto& kv : v.axes) axisValues[kv.first] = kv.second;
    // A virtual press is an edge in this frame
    for (const std::string& name : v.pressed)
    {
        heldNow[name]    = true;
        heldBefore[name] = false;
    }
}

bool InputMap::held(const std::string& name) const
{
    auto it = heldNow.find(name);
    return it != heldNow.end() && it->second;
}

bool InputMap::pressed(const std::string& name) const
{
    auto before = heldBefore.find(name);
    return held(name) && (before == heldBefore.end() || !before->second);
}

double InputMap::axis(const std::string& name) const
{
    auto it = axisValues.find(name);
    return it == axisValues.end() ? 0 : it->second;
}

Vec2 InputMap::move() const
{
    Vec2 m(moveX.empty() ? 0 : axis(moveX), moveZ.empty() ? 0 : axis(moveZ));
    const double n = m.norm();
    return n > 1 ? Vec2(m / n) : m;
}

uint32_t InputMap::heldBits() const
{
    uint32_t bits = 0;
    if (!actionList) return bits;
    for (const auto& h : actionList->held)
        if (held(h.name)) bits |= 1u << h.bit;
    return bits;
}

std::vector<uint32_t> InputMap::pressedActions() const
{
    std::vector<uint32_t> ids;
    if (!actionList) return ids;
    for (const auto& a : actionList->actions)
        if (pressed(a.name)) ids.push_back(a.id);
    return ids;
}

bool InputMap::neutral() const
{
    for (const auto& kv : heldNow)
        if (kv.second) return false;
    for (const auto& kv : axisValues)
        if (kv.second != 0) return false;
    return true;
}

Json InputMap::describe() const
{
    auto names = [](const std::vector<int>& codes)
    {
        Json::array a;
        for (int c : codes)
            for (const auto& kv : keyTable())
                if (kv.second == c)
                {
                    a.push_back(kv.first);
                    break;
                }
        return a;
    };
    Json::object b, x;
    for (const auto& kv : bindings) b[kv.first] = names(kv.second);
    for (const auto& kv : axes)
    {
        Json::object o{{"negative", names(kv.second.negative)}, {"positive", names(kv.second.positive)}};
        if (kv.second.mouseAxis >= 0)
        {
            o["mouse"] = kv.second.mouseAxis == 0 ? "x" : "y";
            o["scale"] = kv.second.scale;
        }
        x[kv.first] = o;
    }
    return Json::object{{"bindings", b}, {"axes", x}, {"move", Json::array{moveX, moveZ}}};
}

}  // namespace Engine
}  // namespace Saiga
