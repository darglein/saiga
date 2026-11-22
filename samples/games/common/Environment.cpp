/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "games/common/Environment.h"

#include "saiga/engine/lighting/EnvironmentFile.h"
#include "saiga/engine/lighting/Systems.h"
#include "saiga/engine/world/SystemList.h"
#include "saiga/saiga_buildconfig.h"

#include <fstream>

namespace Saiga
{
namespace Game
{
namespace
{
// WorldSetup::systems is a function pointer: the game's systems, then the environment's
void (*gameSystems)(Engine::SystemList&) = nullptr;
void systemsWithEnvironment(Engine::SystemList& l)
{
    if (gameSystems) gameSystems(l);
    Engine::addEnvironmentSystems(l);
}
}  // namespace

void enableEnvironment(Engine::EngineAppSetup& setup, const std::string& game)
{
    std::string dir = setup.world.dataDirectory;
    if (dir.empty())
    {
        const std::string source = SAIGA_PROJECT_SOURCE_DIR;
        dir                      = (source.empty() ? std::string("data") : source + "/data") + "/" + game;
    }
    const std::string path = dir + "/environment.json";
    if (std::ifstream(path).good())
    {
        const Engine::EnvironmentFile file = Engine::readEnvironmentFile(path);
        setup.world.environment            = file.setup;
        setup.environment                  = file.settings;
        setup.render.lighting              = file.lighting;
    }
    setup.world.environment.enabled = true;
    setup.render.toneMap            = Vulkan::ToneMap::Aces;
    // Until the S36a1c flip: the S36 light model's pipeline variant
    setup.render.lighting.pbr = true;
    if (setup.world.systems != systemsWithEnvironment)
    {
        gameSystems         = setup.world.systems;
        setup.world.systems = systemsWithEnvironment;
    }
}

}  // namespace Game
}  // namespace Saiga
