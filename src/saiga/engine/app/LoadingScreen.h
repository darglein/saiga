/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The loading screen of a space switch (docs/engine_design.md §12.4 step 2; S21): EngineApp tracks a load when the view
// camera's space changes to a space whose visual data around the camera is not ready (World::spaceReadiness < 1) and
// draws an ImGui screen (an engine default the game can replace, EngineAppSetup::loadingScreen) until it is. Local and
// visual only: the simulation does not wait (before S24). S24: a pending arrival of the local player starts the load at
// the request frame (its island progress), then the switch rules continue it.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

#include <cstddef>
#include <cstdint>

namespace Saiga
{
namespace Engine
{
class EngineApp;

// A running load (EngineApp::spaceLoading)
struct SpaceLoading
{
    SpaceId space = 0;
    // The running maximum of the readiness of the view camera (never moves back when the camera moves)
    double progress = 0;
    // The rendered frames since the load started (the switch frame is 1); S24: of an arrival, since the move
    uint64_t frames = 0;
    // S24 (G19): the load of the local player's arrival, started at the request frame (entity: its root): while the
    // arrival is pending the progress is its island progress scaled to [0, 0.5] (no frame cap: the player is held and
    // hidden); after the move the space's readiness scaled to [0.5, 1], ending when the player's chunk is ready too
    bool arrival = false;
    Entity entity;
};

// End rule 1 and 3 (end rule 2, a complete point, is the app's): the readiness r of this frame is 1 and no terrain
// upload is pending (the sent tiles are recorded, the next frame shows them), or the load ran maxFrames frames
SAIGA_ENGINE_API bool loadEnds(const SpaceLoading& loading, double r, size_t uploadsPending, uint64_t maxFrames);

// The engine's default loading screen (ImGui until S46): an opaque background over the whole viewport (the background
// draw list), the space's name and a progress bar. Called inside the ImGui frame
SAIGA_ENGINE_API void drawLoadingScreen(EngineApp& app, const SpaceLoading& loading);

}  // namespace Engine
}  // namespace Saiga
