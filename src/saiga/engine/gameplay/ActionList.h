/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
/**
 * The game's player actions and held buttons by name (docs/engine_design.md §8.3, §9.3, R-REG-1; S13): one-shot actions
 * are PlayerAction::action codes (ActionEvent), held buttons are bits of PlayerInput::heldButtons. Registered in code
 * (WorldSetup::actions); the input map binds keys to these names (data/<game>/input.json) and the world API accepts the
 * names. The codes are saved as numbers (queued actions): a game never renumbers them.
 */
class ActionList
{
   public:
    struct Action
    {
        uint32_t id;
        std::string name;
    };
    struct Held
    {
        int bit;
        std::string name;
    };

    void add(uint32_t id, const char* name)
    {
        SAIGA_ASSERT(name && *name, "ActionList::add: a name");
        for (const Action& a : actions)
            SAIGA_ASSERT(a.id != id && a.name != name, std::string("ActionList::add: ") + name + " is not unique");
        // One namespace for actions and held buttons (input bindings, set_input)
        for (const Held& h : held)
            SAIGA_ASSERT(h.name != name, std::string("ActionList::add: ") + name + " is a held button");
        actions.push_back(Action{id, name});
    }
    void addHeld(int bit, const char* name)
    {
        SAIGA_ASSERT(name && *name && bit >= 0 && bit < 32, "ActionList::addHeld: a name and a bit 0..31");
        for (const Held& h : held)
            SAIGA_ASSERT(h.bit != bit && h.name != name,
                         std::string("ActionList::addHeld: ") + name + " is not unique");
        for (const Action& a : actions)
            SAIGA_ASSERT(a.name != name, std::string("ActionList::addHeld: ") + name + " is an action");
        held.push_back(Held{bit, name});
    }

    const Action* find(const std::string& name) const
    {
        for (const Action& a : actions)
            if (a.name == name) return &a;
        return nullptr;
    }
    const Held* findHeld(const std::string& name) const
    {
        for (const Held& h : held)
            if (h.name == name) return &h;
        return nullptr;
    }

    std::vector<Action> actions;
    std::vector<Held> held;
};

}  // namespace Engine
}  // namespace Saiga
