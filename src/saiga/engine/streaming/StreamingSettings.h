/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

namespace Saiga
{
namespace Engine
{
// The runtime settings of a streaming world's residency (docs/engine_design.md §12.3; S24): not saved, not game policy
// (World::setStreamingSettings). maxInFlight: residency jobs on the streaming pool at once (the rest wait in the
// residency's ordered queue); synchronous: every requested chunk is built inline in the update that requests it (no
// pool; ScenarioWorld sets it, §8.6)
struct StreamingSettings
{
    int maxInFlight  = 8;
    bool synchronous = false;
};

}  // namespace Engine
}  // namespace Saiga
