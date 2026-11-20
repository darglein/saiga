/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The command line and main of engine games (S13).

#include "saiga/engine/app/EngineMain.h"

#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/tools/ScenarioRunner.h"
#include "saiga/saiga_buildconfig.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace Saiga
{
namespace Engine
{
namespace
{
bool parseInt(const char* s, int min, int max, int& out)
{
    if (!s || !*s) return false;
    char* end    = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (*end != 0 || v < min || v > max) return false;
    out = int(v);
    return true;
}
}  // namespace

bool parseCommandLine(int argc, const char* const* argv, const std::vector<std::string>& gameFlags,
                      EngineAppSetup& setup, AppOptions& options, std::string& error)
{
    if (const char* port = std::getenv("SAIGA_MCP_PORT"))
        if (!parseInt(port, 0, 65535, setup.mcpPort))
        {
            error = "SAIGA_MCP_PORT: a port 0..65535";
            return false;
        }
    if (const char* shot = std::getenv("SAIGA_SCREENSHOT"); shot && *shot) options.screenshot = shot;
    bool dataGiven = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value          = [&](const char* what) -> const char*
        {
            if (i + 1 >= argc || argv[i + 1][0] == '\0')
            {
                error = a + " needs " + what;
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--headless")
            setup.headless = true;
        else if (a == "--no-render")
            setup.headless = setup.noRender = true;
        else if (a == "--validation")
            setup.validation = true;
        else if (a == "--no-vsync")
            setup.vsync = false;
        else if (a == "--benchmark")
        {
            const char* v = value("a name");
            if (!v) return false;
            options.benchmark = v;
        }
        else if (a == "--screenshot")
        {
            const char* v = value("a file");
            if (!v) return false;
            options.screenshot = v;
        }
        else if (a == "--screenshot-frame" || a == "--width" || a == "--height" || a == "--mcp-port")
        {
            const char* v = value("a number");
            int n         = 0;
            if (!v) return false;
            const int max = a == "--mcp-port" ? 65535 : 100000;
            if (!parseInt(v, a == "--mcp-port" ? 0 : 1, max, n))
            {
                error = a + ": a number in range";
                return false;
            }
            if (a == "--screenshot-frame") options.screenshotFrame = n;
            if (a == "--width") setup.width = n;
            if (a == "--height") setup.height = n;
            if (a == "--mcp-port") setup.mcpPort = n;
        }
        else if (a == "--data")
        {
            const char* v = value("a directory");
            if (!v) return false;
            setup.world.dataDirectory = v;
            dataGiven                 = true;
        }
        else if (a == "--scenarios")
        {
            options.runScenarios = true;
            // An optional filter (not another option)
            if (i + 1 < argc && argv[i + 1][0] != '-') options.scenarioFilter = argv[++i];
        }
        else if (std::find(gameFlags.begin(), gameFlags.end(), a) != gameFlags.end())
            options.gameFlags.push_back(a);
        else
        {
            error = "unknown argument '" + a + "'";
            return false;
        }
    }
    if (!options.benchmark.empty() && setup.headless)
    {
        error = "--benchmark needs a window (not --headless)";
        return false;
    }
    if (!dataGiven && setup.world.dataDirectory.empty())
        setup.world.dataDirectory = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/" + setup.name;
    return true;
}

int runGameScenarios(const EngineAppSetup& setup, const std::string& filter)
{
    ScenarioList list;
    if (setup.scenarios) setup.scenarios(list);
    if (list.match(filter).empty())
    {
        std::cout << "no scenarios matched" << (filter.empty() ? "" : " '" + filter + "'") << std::endl;
        return 1;
    }
    const std::vector<ScenarioResult> results = runScenarios(setup.world, list, filter);
    int failed                                = 0;
    for (const ScenarioResult& r : results)
    {
        std::cout << (r.passed ? "passed " : "FAILED ") << r.scenario << " (" << variantName(r.variant) << ", "
                  << r.steps << " steps, " << r.seconds << " s)" << std::endl;
        if (!r.passed)
        {
            failed++;
            std::cout << r.report;
        }
    }
    std::cout << results.size() - size_t(failed) << " passed, " << failed << " failed" << std::endl;
    return failed == 0 ? 0 : 1;
}

int engineMain(int argc, const char* const* argv, EngineAppSetup setup, const std::vector<std::string>& gameFlags,
               const std::function<void(EngineApp&, const AppOptions&)>& start)
{
    AppOptions options;
    std::string error;
    if (!parseCommandLine(argc, argv, gameFlags, setup, options, error))
    {
        std::cerr << error << std::endl;
        return 1;
    }
    if (options.runScenarios) return runGameScenarios(setup, options.scenarioFilter);
    EngineApp app(setup, options);
    if (start) start(app, options);
    return app.run();
}

}  // namespace Engine
}  // namespace Saiga
