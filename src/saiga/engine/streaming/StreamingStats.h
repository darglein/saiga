/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace Saiga
{
namespace Engine
{
// The statistics of a world's streaming (docs/engine_design.md §12.3; S24): ChunkResidency::stats(), streaming_status,
// world_status, the Engine tab, the benchmark. Not state (not saved, not hashed)
struct StreamingStats
{
    // ---- cumulative
    // Chunks requested, residency results applied, results dropped (an old request serial or placement version),
    // results applied after the chunk's terrain sequence changed since the request (G6), chunks unloaded
    uint64_t requested = 0, applied = 0, resultsDropped = 0, appliedAfterModification = 0, unloaded = 0;
    // Chunks built inline (synchronous mode), steps with a player on a chunk that is not ready on this peer and the sum
    // of such players over the steps (G11), steps above maxSimulationIslands (G14), entities held by the mover
    uint64_t syncBuilt = 0, streamingStalls = 0, playerStallSteps = 0, islandLimitSteps = 0, moverHolds = 0;
    // Root evaluations of the Simulated tag
    uint64_t evaluations = 0;
    // ---- current
    int centres = 0, requestedChunks = 0, residentChunks = 0, pendingJobs = 0, queuedChunks = 0, islands = 0;
    int simulated          = 0;
    size_t residentBytes   = 0;
    double lastStreamingMs = 0, lastSimulationMs = 0, lastLoadRebuildMs = 0;
};

}  // namespace Engine
}  // namespace Saiga
