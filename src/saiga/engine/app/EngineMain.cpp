/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The command line and main of engine games (S13).

#include "saiga/engine/app/EngineMain.h"

#include "saiga/core/util/crash.h"
#include "saiga/core/util/env.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/tools/ScenarioRunner.h"
#include "saiga/saiga_buildconfig.h"

#include <algorithm>
#include <cmath>
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
    if (auto port = GetEnvIfSet("SAIGA_MCP_PORT"))
        if (!parseInt(port->c_str(), 0, 65535, setup.mcpPort))
        {
            error = "SAIGA_MCP_PORT: a port 0..65535";
            return false;
        }
    if (auto shot = GetEnvIfSet("SAIGA_SCREENSHOT"); shot && !shot->empty()) options.screenshot = *shot;
    if (auto tab = GetEnvIfSet("SAIGA_DEBUG_TAB"); tab && !tab->empty())
        options.debugTab = std::max(std::atoi(tab->c_str()), 0);
    bool dataGiven = false, screenshotFromArgument = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto value          = [&](const char* what) -> const char*
        {
            // An option is not a value (--data --headless)
            if (i + 1 >= argc || argv[i + 1][0] == '\0' || std::strncmp(argv[i + 1], "--", 2) == 0)
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
        else if (a == "--stats")
            options.showStatistics = true;
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
            options.screenshot     = v;
            screenshotFromArgument = true;
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
        else if (a == "--tonemap")
        {
            const char* v = value("identity or aces");
            if (!v) return false;
            const std::string t = v;
            if (t == "identity")
                setup.render.toneMap = Vulkan::ToneMap::Identity;
            else if (t == "aces")
                setup.render.toneMap = Vulkan::ToneMap::Aces;
            else
            {
                error = "--tonemap: identity or aces";
                return false;
            }
        }
        else if (a == "--exposure")
        {
            const char* v = value("a number");
            if (!v) return false;
            char* end     = nullptr;
            const float x = std::strtof(v, &end);
            // Positive and finite (RenderScene clamps it to [1/1024, 1024])
            if (*end != 0 || !(x > 0) || !std::isfinite(x))
            {
                error = "--exposure: a positive number";
                return false;
            }
            setup.render.exposure = x;
        }
        else if (a == "--time-of-day" || a == "--day-length")
        {
            // S36: WorldTime's hours [0, 24), or the real seconds of a game day (rate = 86400 / s, at most 3600)
            const char* v = value("a number");
            if (!v) return false;
            char* end      = nullptr;
            const double x = std::strtod(v, &end);
            const bool tod = a == "--time-of-day";
            if (*end != 0 || !std::isfinite(x) || (tod ? !(x >= 0 && x < 24) : !(x >= 86400.0 / kMaxTimeRate)))
            {
                error = tod ? "--time-of-day: hours in [0, 24)" : "--day-length: seconds of a game day, at least 24";
                return false;
            }
            (tod ? options.timeOfDay : options.dayLength) = x;
        }
        else if (a == "--data")
        {
            const char* v = value("a directory");
            if (!v) return false;
            setup.world.dataDirectory = v;
            dataGiven                 = true;
        }
        else if (a == "--save")
        {
            const char* v = value("a directory");
            if (!v) return false;
            setup.saveDirectory = v;
        }
        else if (a == "--coop")
        {
            options.coop = true;
            // An optional delay in steps: an integer argument is consumed (0..30), anything else is not
            if (i + 1 < argc)
            {
                const char* next = argv[i + 1];
                char* end        = nullptr;
                const long v     = std::strtol(next, &end, 10);
                if (*next != 0 && *end == 0)
                {
                    if (v < 0 || v > 30)
                    {
                        error = "--coop: the delay is 0..30 steps";
                        return false;
                    }
                    options.coopDelay = int(v);
                    ++i;
                }
            }
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
    if (!options.benchmark.empty())
    {
        if (setup.headless)
        {
            error = "--benchmark needs a window (not --headless)";
            return false;
        }
        if (std::find(setup.benchmarks.begin(), setup.benchmarks.end(), options.benchmark) == setup.benchmarks.end())
        {
            error = "unknown benchmark '" + options.benchmark + "'";
            if (!setup.benchmarks.empty())
            {
                error += " (";
                for (size_t k = 0; k < setup.benchmarks.size(); ++k) error += (k ? ", " : "") + setup.benchmarks[k];
                error += ")";
            }
            return false;
        }
    }
    if (options.coop && (options.runScenarios || !options.benchmark.empty()))
    {
        error = options.runScenarios ? "--coop with --scenarios (the scenarios run their own host + client variants)"
                                     : "--coop with --benchmark";
        return false;
    }
    if (!setup.saveDirectory.empty() && (options.runScenarios || !options.benchmark.empty()))
    {
        // The scenarios and the benchmark build or replace their own worlds
        error = std::string("--save with ") + (options.runScenarios ? "--scenarios" : "--benchmark");
        return false;
    }
    if (options.runScenarios && (options.timeOfDay || options.dayLength))
    {
        // The scenarios build their own worlds: the options would be ignored silently
        error = std::string(options.timeOfDay ? "--time-of-day" : "--day-length") + " with --scenarios";
        return false;
    }
    if (setup.checkOptions && !setup.checkOptions(options, error)) return false;
    if (setup.noRender && !options.screenshot.empty())
    {
        if (screenshotFromArgument)
        {
            error = "--screenshot needs rendering (not --no-render)";
            return false;
        }
        std::cerr << "SAIGA_SCREENSHOT is ignored with --no-render" << std::endl;
        options.screenshot.clear();
    }
    if (!dataGiven && setup.world.dataDirectory.empty())
    {
        // The source tree's data (SAIGA_WRITE_PROJECT_DIR_TO_HEADER), else data/<name> next to the working directory
        const std::string source  = SAIGA_PROJECT_SOURCE_DIR;
        setup.world.dataDirectory = (source.empty() ? std::string("data") : source + "/data") + "/" + setup.name;
    }
    if (setup.onDataDirectory) setup.onDataDirectory(setup);
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
    ScenarioOptions scenarioOptions;
    scenarioOptions.networkVariants           = setup.networkVariants;
    scenarioOptions.directoryRoundTrip        = false;
    const std::vector<ScenarioResult> results = runScenarios(setup.world, list, filter, scenarioOptions);
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
    catchSegFaults();
    if (!parseCommandLine(argc, argv, gameFlags, setup, options, error))
    {
        std::cerr << error << "\nUsage: " << setup.name
                  << " [--headless] [--no-render] [--benchmark <name>] [--screenshot <file>] [--screenshot-frame N] "
                     "[--width N] [--height N] [--validation] [--no-vsync] [--stats] [--tonemap identity|aces] "
                     "[--exposure x] [--time-of-day h] [--day-length s] "
                     "[--scenarios [filter]] [--mcp-port N] [--data <dir>] [--coop [delay]] [--save <dir>]";
        for (const std::string& f : gameFlags) std::cerr << " [" << f << "]";
        std::cerr << std::endl;
        return 1;
    }
    bool constructed = false;
    try
    {
        if (options.runScenarios) return runGameScenarios(setup, options.scenarioFilter);
        EngineApp app(setup, options);
        constructed = true;
        if (start) start(app, options);
        // S25: the save directory after the game's start (an existing one replaces the world through the load path)
        app.openSaveDirectory();
        // S36: --time-of-day / --day-length after the game's start (its world exists and has its state)
        if (options.timeOfDay || options.dayLength)
        {
            const char* what = options.timeOfDay && options.dayLength ? "--time-of-day / --day-length"
                               : options.timeOfDay                    ? "--time-of-day"
                                                                      : "--day-length";
            WorldTimeChange change;
            change.hours = options.timeOfDay;
            if (options.dayLength) change.rate = 86400.0 / *options.dayLength;
            if (!hasWorldTime(app.startWorld()))
            {
                std::cerr << what << ": this game has no world time (EnvironmentSetup::enabled)" << std::endl;
                return 1;
            }
            if (!setWorldTime(app.startWorld(), change, error))
            {
                std::cerr << what << ": " << error << std::endl;
                return 1;
            }
        }
        return app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << setup.name << ": " << e.what() << std::endl;
        // The hint only for a failed start with a window or a device (a display or a GPU may be missing)
        if (!constructed && !options.runScenarios && !setup.noRender)
            std::cerr << "(without a display or GPU: --headless renders offscreen, --no-render does not render)"
                      << std::endl;
        return 1;
    }
}

}  // namespace Engine
}  // namespace Saiga
