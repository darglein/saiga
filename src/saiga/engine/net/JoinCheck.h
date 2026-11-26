/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The checks of the handshake (docs/engine_design.md §13.2; S26, D2, D5, D6): the client's JoinRequest of a world, the
// host's check of a request and the client's validation of the host's generation info (the pins it adopts, E11).

#pragma once

#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/net/JoinMessages.h"
#include "saiga/export.h"

#include <optional>
#include <string>
#include <utility>

namespace Saiga
{
namespace Engine
{
class World;

// The JoinRequest of a client world: its schema hash and sections, render table hash, seed, environment flag, the
// non-legacy flag of its generation and the registered generators with version() and slot names
SAIGA_ENGINE_API JoinRequest joinRequestOf(const World& client);
// Host: the first failing check of a request, in the order protocol, environment, schema, render tables, seed,
// generators (D5); nullopt: admit
SAIGA_ENGINE_API std::optional<std::pair<JoinRejectReason, std::string>> checkJoinRequest(const World& host,
                                                                                          const JoinRequest& r);
// Client: validates the host's generation info against this build (the seed, the generator names, pinned <= current
// <= version(), the host's slot table a prefix of this build's) and returns the pins to adopt; false with the error
SAIGA_ENGINE_API bool adoptablePins(const World& client, const GenerationInfo& host, GenerationPins& pins,
                                    std::string& error);

}  // namespace Engine
}  // namespace Saiga
