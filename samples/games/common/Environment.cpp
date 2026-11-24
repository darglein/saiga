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

#include <filesystem>

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
    // The environment file of the final data directory (--data; CodeRabbit #83: it was read before the command line,
    // from the default directory), after the command line's settings (--tonemap, --exposure are other fields)
    setup.onDataDirectory = [game](Engine::EngineAppSetup& s)
    {
        std::string dir = s.world.dataDirectory;
        if (dir.empty())
        {
            const std::string source = SAIGA_PROJECT_SOURCE_DIR;
            dir                      = (source.empty() ? std::string("data") : source + "/data") + "/" + game;
        }
        const std::string path = dir + "/environment.json";
        // Only an absent file is optional (CodeRabbit #83: an unreadable one was skipped too); any other file, or an
        // error of the file system, goes to readEnvironmentFile, which reports it
        std::error_code ec;
        if (std::filesystem::status(path, ec).type() != std::filesystem::file_type::not_found)
        {
            const Engine::EnvironmentFile file = Engine::readEnvironmentFile(path);
            s.world.environment                = file.setup;
            s.environment                      = file.settings;
            s.render.lighting                  = file.lighting;
        }
        s.world.environment.enabled = true;
        // Until the S36a1c flip: the S36 light model's pipeline variant
        s.render.lighting.pbr = true;
    };
    setup.world.environment.enabled = true;
    setup.render.toneMap            = Vulkan::ToneMap::Aces;
    if (setup.world.systems != systemsWithEnvironment)
    {
        gameSystems         = setup.world.systems;
        setup.world.systems = systemsWithEnvironment;
    }
}

}  // namespace Game
}  // namespace Saiga
