/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// A second translation unit of test_engine_world: a DamageEvent in its own unnamed namespace, with the C++ name, event
// ID, name and layout of the one in test_engine_world.cpp (GCC gives both the same typeid name)

#include "saiga/engine/world/World.h"

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
struct DamageEvent
{
    static constexpr const char* name = "DamageEvent";
    static constexpr EventId id       = EventId(64);
    static constexpr uint32_t version = 1;
    Entity target;
    double amount = 0;
    template <class V>
    void visit(V& v)
    {
        v("target", target);
        v("amount", amount);
    }
};
}  // namespace

size_t readDamageEventOfTheSecondTranslationUnit(World& w)
{
    return w.events.read<DamageEvent>().size();
}
