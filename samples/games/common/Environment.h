/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/app/EngineAppSetup.h"

#include <string>

namespace Saiga
{
namespace Game
{
/**
 * The environment of a tech demo (S36a1c, A10 of docs/stages/S36_lighting.plan.md): enables WorldTime and Weather
 * (EnvironmentSetup::enabled), reads data/<game>/environment.json if it exists (R-DATA-1: the celestial model,
 * exposure, ambient, clouds, atmosphere and lighting parameters; racing and the platformer have none and use the code
 * defaults) into the world setup, the app's EnvironmentSettings and LightingParameters, registers addEnvironmentSystems
 * (the time passes with WorldTime::rate) and renders ACES by default (D22: the time of day exists only in the
 * tone-mapped path;
 * --tonemap identity still gives the clay view, parsed after this). Call it before engineMain, after setup.world is
 * set. The file is read when the command line has resolved the data directory (EngineAppSetup::onDataDirectory: --data
 * selects it), else from the source tree's data/<game>.
 */
void enableEnvironment(Engine::EngineAppSetup& setup, const std::string& game);

}  // namespace Game
}  // namespace Saiga
