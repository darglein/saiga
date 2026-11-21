/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The debug UI's logic without ImGui or a device (docs/engine_design.md §17.1; S19c): the console's commands, the tab
// keys, the saved state and the MCP call log

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/DebugConsole.h"
#include "saiga/engine/app/DebugUi.h"
#include "saiga/engine/app/McpCallLog.h"
#include "saiga/engine/core/Log.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <fstream>

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

DebugConsole::Command parse(const std::string& line)
{
    DebugConsole::Command c;
    std::string error;
    EXPECT_TRUE(DebugConsole::parse(line, c, error)) << line << ": " << error;
    return c;
}

std::string parseError(const std::string& line)
{
    DebugConsole::Command c;
    std::string error;
    EXPECT_FALSE(DebugConsole::parse(line, c, error)) << line;
    return error;
}

// A server with an answering tool, a failing tool and a tool that answers later (deferred)
struct TestServer
{
    McpServer server{"test"};
    McpServer::Respond later;
    TestServer()
    {
        server.addTool("echo", "Returns its arguments.",
                       McpServer::ObjectSchema({{"x", McpServer::NumberProperty("a number")}}),
                       [](const Json& a, McpServer::Respond r) { r(McpServer::ToolResult::Text(a.dump())); });
        server.addTool("fail", "Fails.", McpServer::ObjectSchema({}),
                       [](const Json&, McpServer::Respond r) { r(McpServer::ToolResult::Error("it failed")); });
        server.addTool("later", "Answers later.", McpServer::ObjectSchema({}),
                       [this](const Json&, McpServer::Respond r) { later = std::move(r); });
        server.addTool("set_component", "Sets.", McpServer::ObjectSchema({}),
                       [](const Json&, McpServer::Respond r) { r(McpServer::ToolResult::Text("ok")); });
    }
};

InputState keys(std::initializer_list<int> down)
{
    InputState s;
    for (int k : down) s.keys[size_t(k)] = true;
    return s;
}

std::filesystem::path tempFile(const std::string& name)
{
    return std::filesystem::path(::testing::TempDir()) /
           (name + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
}
}  // namespace

TEST(DebugConsole, ParsesCommands)
{
    EXPECT_EQ(parseError("   "), "empty command");
    DebugConsole::Command c = parse("world_status");
    EXPECT_EQ(c.name, "world_status");
    EXPECT_FALSE(c.builtin);
    EXPECT_EQ(c.arguments, Json(Json::object{}));

    c = parse(R"(spawn {"class": "car", "position": [0, 1, 2]})");
    EXPECT_EQ(c.name, "spawn");
    EXPECT_EQ(c.arguments["class"], Json("car"));
    EXPECT_EQ(c.arguments["position"][2], Json(2));

    // key=value: JSON when it parses, else a string; quotes and brackets keep their spaces
    c = parse(R"(spawn class=car n=3 on=true position=[0, 1, 2] label="a b" dt=0.5)");
    EXPECT_EQ(c.arguments["class"], Json("car"));
    EXPECT_EQ(c.arguments["n"], Json(3));
    EXPECT_EQ(c.arguments["on"], Json(true));
    EXPECT_EQ(c.arguments["position"].array_items().size(), 3u);
    EXPECT_EQ(c.arguments["label"], Json("a b"));
    EXPECT_EQ(c.arguments["dt"], Json(0.5));

    EXPECT_NE(parseError("spawn {\"class\": "), "");
    EXPECT_NE(parseError("spawn {} extra").find("not a JSON object"), std::string::npos);
    EXPECT_NE(parseError("spawn [1, 2]").find("key=value"), std::string::npos);
    EXPECT_NE(parseError("spawn class").find("key=value"), std::string::npos);
    EXPECT_NE(parseError("spawn =3").find("key=value"), std::string::npos);
    EXPECT_NE(parseError("spawn label=\"open"), "");
    EXPECT_NE(parseError("spawn p=[1, 2"), "");

    c = parse("help spawn");
    EXPECT_TRUE(c.builtin);
    EXPECT_EQ(c.builtinArgument, "spawn");
    EXPECT_TRUE(parse("tools").builtin);
    EXPECT_TRUE(parse("clear").builtin);
}

TEST(DebugConsole, CompletesToolNames)
{
    TestServer t;
    LogRing ring;
    DebugConsole console(&t.server, ring);
    EXPECT_EQ(console.complete("ec"), (std::vector<std::string>{"echo"}));
    EXPECT_EQ(console.complete("set"), (std::vector<std::string>{"set_component"}));
    // Tools and built-ins, sorted
    EXPECT_EQ(console.complete("l"), (std::vector<std::string>{"later"}));
    EXPECT_EQ(console.complete("c"), (std::vector<std::string>{"clear"}));
    EXPECT_EQ(console.complete("xyz"), std::vector<std::string>{});
    EXPECT_EQ(console.complete("").size(), 7u);
}

TEST(DebugConsole, RunsToolsThroughThePoll)
{
    TestServer t;
    LogRing ring;
    ring.add("app", "started");
    DebugConsole console(&t.server, ring);
    // The log lines that exist already
    ASSERT_EQ(console.entries().size(), 1u);
    EXPECT_EQ(console.entries()[0].text, "started");

    console.submit("echo x=2");
    // Queued: runs only in poll()
    ASSERT_EQ(console.entries().size(), 2u);
    EXPECT_TRUE(console.entries()[1].pending);
    EXPECT_EQ(console.pendingCalls(), 1);
    t.server.poll();
    ASSERT_EQ(console.entries().size(), 3u);
    EXPECT_FALSE(console.entries()[1].pending);
    EXPECT_EQ(console.entries()[2].kind, DebugConsole::Entry::Kind::Output);
    EXPECT_EQ(console.entries()[2].text, R"({"x": 2})");
    EXPECT_EQ(console.pendingCalls(), 0);

    console.submit("fail");
    t.server.poll();
    EXPECT_EQ(console.entries().back().kind, DebugConsole::Entry::Kind::Error);
    EXPECT_EQ(console.entries().back().text, "it failed");

    // Unknown tools and parse errors
    console.submit("nope");
    t.server.poll();
    EXPECT_EQ(console.entries().back().kind, DebugConsole::Entry::Kind::Error);
    EXPECT_NE(console.entries().back().text.find("Unknown tool"), std::string::npos);
    console.submit("echo x");
    EXPECT_EQ(console.entries().back().kind, DebugConsole::Entry::Kind::Error);

    // A deferred answer arrives in a later poll; lines logged meanwhile come first
    console.submit("later");
    t.server.poll();
    EXPECT_EQ(console.pendingCalls(), 1);
    ASSERT_TRUE(t.later);
    ring.add("app", "meanwhile");
    console.update();
    t.later(McpServer::ToolResult::Text("done"));
    EXPECT_EQ(console.entries().back().text, "meanwhile");
    t.server.poll();
    EXPECT_EQ(console.entries().back().text, "done");
    EXPECT_EQ(console.pendingCalls(), 0);

    // The answers are in the console only, not in the log ring
    for (const auto& l : ring.since(0)) EXPECT_EQ(l.category, "app") << l.message;
    // The history (no repeated entries)
    console.submit("later");
    EXPECT_EQ(console.history, (std::vector<std::string>{"echo x=2", "fail", "nope", "echo x", "later"}));
    t.server.poll();
    t.later(McpServer::ToolResult::Text("done"));
    t.server.poll();
}

TEST(DebugConsole, BuiltinsAndFilters)
{
    TestServer t;
    LogRing ring;
    DebugConsole console(&t.server, ring);
    console.submit("tools");
    EXPECT_NE(console.entries().back().text.find("echo  fail  later"), std::string::npos);
    console.submit("help echo");
    EXPECT_NE(console.entries().back().text.find("x? (number): a number"), std::string::npos);
    console.submit("help nope");
    EXPECT_EQ(console.entries().back().kind, DebugConsole::Entry::Kind::Error);
    console.submit("clear");
    EXPECT_TRUE(console.entries().empty());

    ring.add("mcp", "echo -> ok");
    ring.add("app", "Error: something");
    console.update();
    ASSERT_EQ(console.entries().size(), 2u);
    EXPECT_EQ(console.categories(), (std::set<std::string>{"app", "mcp"}));
    console.mcpOnly = true;
    EXPECT_TRUE(console.visible(console.entries()[0]));
    EXPECT_FALSE(console.visible(console.entries()[1]));
    console.mcpOnly = false;
    console.hiddenCategories.insert("mcp");
    EXPECT_FALSE(console.visible(console.entries()[0]));
    console.hiddenCategories.clear();
    console.search = "SOMETHING";
    EXPECT_FALSE(console.visible(console.entries()[0]));
    EXPECT_TRUE(console.visible(console.entries()[1]));
}

// Regression guard: an answer that arrives after the console is gone must not touch it
TEST(DebugConsole, AnswerAfterTheConsoleIsGone)
{
    TestServer t;
    LogRing ring;
    {
        DebugConsole console(&t.server, ring);
        console.submit("later");
        t.server.poll();
    }
    ASSERT_TRUE(t.later);
    t.later(McpServer::ToolResult::Text("late"));
    t.server.poll();
}

TEST(DebugUi, KeysSelectTabs)
{
    DebugUi ui;
    const int F1 = DebugUi::kKeyF1, grave = DebugUi::kKeyGrave;
    EXPECT_EQ(ui.selected(), int(DebugTab::Game));
    // A press selects; held is one press; the selected tab's key changes nothing (tabs, not toggles)
    InputState out = ui.handleKeys(keys({F1 + 1}), keys({}), false);
    EXPECT_TRUE(ui.isSelected(DebugTab::Performance));
    EXPECT_FALSE(out.keys[size_t(F1 + 1)]);  // the input map never sees F1..F10
    ui.select(DebugTab::Engine);
    ui.handleKeys(keys({F1 + 1}), keys({F1 + 1}), false);
    EXPECT_TRUE(ui.isSelected(DebugTab::Engine));
    ui.handleKeys(keys({F1 + 2}), keys({}), false);
    ui.handleKeys(keys({F1 + 2}), keys({F1 + 2}), false);
    EXPECT_TRUE(ui.isSelected(DebugTab::Engine));
    // Also while ImGui has the keyboard (a focused console line)
    ui.handleKeys(keys({F1}), keys({}), true);
    EXPECT_TRUE(ui.isSelected(DebugTab::Game));
    // The grave key: the console, only while not captured (it types into a focused line)
    ui.handleKeys(keys({grave}), keys({}), true);
    EXPECT_TRUE(ui.isSelected(DebugTab::Game));
    out = ui.handleKeys(keys({grave, 'W'}), keys({}), false);
    EXPECT_TRUE(ui.isSelected(DebugTab::Console));
    EXPECT_TRUE(out.keys[size_t('W')]);
    // A game tab gets the next key; keys without a tab do nothing
    EXPECT_EQ(ui.addTab("AI", {}), 4);
    EXPECT_EQ(ui.tabKey(4), 5);
    ui.handleKeys(keys({F1 + 4}), keys({}), false);
    EXPECT_EQ(ui.selected(), 4);
    ui.handleKeys(keys({F1 + 9}), keys({}), false);
    EXPECT_EQ(ui.selected(), 4);
    EXPECT_EQ(ui.findTab("ai"), 4);
    EXPECT_EQ(ui.findTab("PERFORMANCE"), 1);
    EXPECT_EQ(ui.findTab("nope"), -1);
    // Disabled: the keys pass through
    ui.setEnabled(false);
    out = ui.handleKeys(keys({F1 + 1}), keys({}), false);
    EXPECT_EQ(ui.selected(), 4);
    EXPECT_TRUE(out.keys[size_t(F1 + 1)]);
}

TEST(DebugUi, StatePersists)
{
    const std::filesystem::path file = tempFile("debug_ui_state");
    {
        DebugUi ui(nullptr, file.string());
        ui.load();  // no file yet: the defaults
        EXPECT_TRUE(ui.isSelected(DebugTab::Game));
        ui.select(DebugTab::Console);
        Json s           = ui.state();
        Json::object o   = s.object_items();
        o["panel_width"] = 480;
        o["sections"]    = Json::object{{"engine/Systems", true}, {"performance/Graph", false}};
        ui.setState(o);
        ui.console().history = {"world_status", "spawn class=car"};
        ui.console().hiddenCategories.insert("api");
        ui.console().mcpOnly = true;
        ui.save();
    }
    ASSERT_TRUE(std::filesystem::exists(file));
    {
        DebugUi ui(nullptr, file.string());
        ui.load();
        EXPECT_TRUE(ui.isSelected(DebugTab::Console));
        const Json s = ui.state();
        EXPECT_EQ(s["panel_width"].number_value(), 480);
        EXPECT_EQ(s["sections"]["engine/Systems"], Json(true));
        EXPECT_EQ(s["sections"]["performance/Graph"], Json(false));
        EXPECT_EQ(ui.console().history, (std::vector<std::string>{"world_status", "spawn class=car"}));
        EXPECT_EQ(ui.console().hiddenCategories, (std::set<std::string>{"api"}));
        EXPECT_TRUE(ui.console().mcpOnly);
    }
    // Invalid JSON: the defaults
    {
        std::ofstream(file) << "{ not json";
        DebugUi ui(nullptr, file.string());
        ui.load();
        EXPECT_TRUE(ui.isSelected(DebugTab::Game));
    }
    std::filesystem::remove(file);
    // Without a file name nothing is read or written
    {
        DebugUi ui;
        ui.select(DebugTab::Engine);
        ui.save();
        EXPECT_EQ(ui.stateFile(), "");
    }
}

TEST(McpCallLog, LinesAndThrottling)
{
    double now = 0;
    std::vector<std::string> lines;
    McpCallLog log([&]() { return now; }, [&](const std::string& l) { lines.push_back(l); });
    using R     = McpServer::CallRecord;
    auto record = [&](uint64_t id, const std::string& tool, R::Phase phase, R::Source source = R::Source::Remote,
                      bool error = false)
    {
        R r;
        r.id         = id;
        r.tool       = tool;
        r.arguments  = tool == "spawn" ? Json(Json::object{{"class", "car"}}) : Json(Json::object{});
        r.phase      = phase;
        r.source     = source;
        r.isError    = error;
        r.durationMs = 1.25;
        r.answer     = error ? "no such class" : "{\"entity\": 3}";
        log.record(r);
    };
    record(1, "spawn", R::Phase::Started);
    record(1, "spawn", R::Phase::Answered);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], R"(spawn -> ok 1.2 ms: {"entity": 3}  args {"class": "car"})");
    record(2, "spawn", R::Phase::Answered, R::Source::Local, true);
    EXPECT_EQ(lines.back(), R"((console) spawn -> error 1.2 ms: no such class  args {"class": "car"})");

    // Polling tools: at most one line per 2 s, the calls in between counted
    lines.clear();
    record(3, "world_status", R::Phase::Answered);
    now = 0.5;
    record(4, "world_status", R::Phase::Answered);
    now = 1.0;
    record(5, "world_status", R::Phase::Answered);
    EXPECT_EQ(lines.size(), 1u);
    log.tick();
    EXPECT_EQ(lines.size(), 1u);
    now = 2.1;
    log.tick();
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[1], "world_status: +2 calls");
    now = 2.5;
    record(6, "world_status", R::Phase::Answered);
    now = 4.2;
    record(7, "world_status", R::Phase::Answered);
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_NE(lines[2].find("(+1 calls)"), std::string::npos) << lines[2];
    // The console's polling calls are not throttled
    record(8, "world_status", R::Phase::Answered, R::Source::Local);
    EXPECT_EQ(lines.size(), 4u);

    // A long call: a running line after 0.5 s, and the longest running call for the bar (not screenshot)
    lines.clear();
    now = 10;
    record(9, "run_scenario", R::Phase::Started);
    record(10, "screenshot", R::Phase::Started);
    now = 10.3;
    log.tick();
    EXPECT_TRUE(lines.empty());
    now = 10.6;
    log.tick();
    log.tick();
    // Once per call
    EXPECT_EQ(lines, (std::vector<std::string>{"run_scenario ... running", "screenshot ... running"}));
    double seconds = 0;
    EXPECT_EQ(log.longestRunning(&seconds), "run_scenario");
    EXPECT_NEAR(seconds, 0.6, 1e-9);
    record(9, "run_scenario", R::Phase::Answered);
    EXPECT_EQ(log.longestRunning(), "");
}

TEST(McpCallLog, NotEchoedByDefault)
{
    // SAIGA_MCP_ECHO=1 turns the echo of the "mcp" lines on; the tests run without it
    if (std::getenv("SAIGA_MCP_ECHO")) GTEST_SKIP() << "SAIGA_MCP_ECHO is set";
    EXPECT_FALSE(mcpLogEcho());
    // Through the default sink into the engine log, category mcp
    const uint64_t before = engineLog().lastSequence();
    McpCallLog log;
    McpServer::CallRecord r;
    r.tool  = "quit";
    r.phase = McpServer::CallRecord::Phase::Answered;
    log.record(r);
    const auto lines = engineLog().since(before);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0].category, "mcp");
    EXPECT_EQ(lines[0].message, "quit -> ok 0.0 ms");
}

// Regression (review): the saved tab or SAIGA_DEBUG_TAB naming a game tab was ignored, because the state is read before
// the game adds its tabs (and the file was then saved with "Game")
TEST(DebugUi, GameTabSelectedWhenItIsAdded)
{
    DebugUi ui;
    ui.setState(Json::object{{"tab", "AI"}});
    EXPECT_TRUE(ui.isSelected(DebugTab::Game));
    EXPECT_EQ(ui.state()["tab"], Json("AI"));
    ui.addTab("Physics", {});
    EXPECT_TRUE(ui.isSelected(DebugTab::Game));
    EXPECT_EQ(ui.addTab("AI", {}), 5);
    EXPECT_EQ(ui.selected(), 5);
    EXPECT_EQ(ui.state()["tab"], Json("AI"));

    DebugUi byKey;
    byKey.select(4);  // SAIGA_DEBUG_TAB=5
    byKey.addTab("AI", {});
    EXPECT_EQ(byKey.selected(), 4);
}
