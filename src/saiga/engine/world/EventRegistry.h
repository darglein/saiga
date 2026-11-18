/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/world/EventQueues.h"

namespace Saiga
{
namespace Engine
{
/**
 * Registration of event types (docs/engine_design.md §7.7, §7.9): registerEngineEvents(EventRegistry&) and
 * WorldSetup::events. An event is a struct with name, id (EventId: EngineEvent 0..63, game enum from 64), version and
 * visit (the field rules of components, §7.2).
 */
class EventRegistry
{
   public:
    explicit EventRegistry(EventQueues& queues) : queues(queues) {}

    template <class E>
    void add(Replicate mode = Replicate::Never)
    {
        queues.registerEvent<E>(mode);
    }

   private:
    EventQueues& queues;
};

}  // namespace Engine
}  // namespace Saiga
