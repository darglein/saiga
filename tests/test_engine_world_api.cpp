/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The JSON visitors, the world API and its MCP tools (docs/engine_design.md §16.2; stage S10)

#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/VisitJson.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/tools/WorldTools.h"
#include "saiga/engine/world/World.h"
#include "saiga/saiga_buildconfig.h"

#include "gtest/gtest.h"

#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mcp_test_helper.h"

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
// The api log lines of the edits are not echoed to stdout in the tests
const bool g_noEcho = (setLogEcho(false), true);

enum class TestClass : uint16_t
{
    Dummy = kFirstGameClass,
    Count
};
enum class TestComponent : uint8_t
{
    Weather = kFirstGameComponent,
    Count
};

struct Weather
{
    static constexpr const char* name = "Weather";
    static constexpr ComponentId id   = ComponentId(TestComponent::Weather);
    static constexpr uint32_t version = 1;
    double rain                       = 0;
    template <class V>
    void visit(V& v)
    {
        v("rain", rain);
    }
};

// Takes damage through its rules (half of it); records actions in Health::maxHp
class Dummy final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Dummy);
    static constexpr const char* name = "Dummy";
    void onSpawn() override
    {
        if (!has<Health>()) add<Health>();
    }
    void onDamage(const DamageEvent& e) override { write<Health>().hp -= e.amount / 2; }
};

// The data type of data/tests/engine/creature_stats.json
struct CreatureStats
{
    static constexpr const char* name = "CreatureStats";
    static constexpr uint32_t version = 1;
    double speed                      = 1;
    double health                     = 10;
    std::string sound;
    template <class V>
    void visit(V& v)
    {
        v("speed", speed);
        v("health", health);
        v("sound", sound);
    }
};

std::vector<uint32_t> g_actions;
void sysActions(World& w, double)
{
    for (const ActionEvent& a : w.events.read<ActionEvent>()) g_actions.push_back(a.action);
}

WorldSetup apiSetup()
{
    WorldSetup s;
    s.components    = [](Registry& r) { r.registerGlobal<Weather>(Replicate::Host); };
    s.classes       = [](ClassList& c) { c.add<Dummy>(); };
    s.systems       = [](SystemList& l) { l.add(Phase::Gameplay, "actions", sysActions); };
    s.dataTypes     = [](DataTypeList& d) { d.add<CreatureStats>(); };
    s.dataDirectory = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine";
    return s;
}

Json obj(const std::string& text)
{
    std::string err;
    Json j = Json::parse(text, err);
    EXPECT_TRUE(err.empty()) << err << " in " << text;
    return j;
}

Entity entityOf(const Json& j)
{
    return Entity{uint32_t(j["index"].number_value()), uint32_t(j["generation"].number_value())};
}

std::string entityString(Entity e)
{
    return "\"" + std::to_string(e.index) + ":" + std::to_string(e.generation) + "\"";
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// JSON visitors

TEST(VisitJson, RoundTripOfEveryEngineComponentType)
{
    Player p;
    p.peer        = 3;
    p.characterId = (uint64_t(1) << 60) + 5;  // beyond 2^53: a string
    p.moveSpeed   = 7.5;
    Json j        = toJson(p);
    EXPECT_TRUE(j["characterId"].is_string());
    Player q;
    ASSERT_TRUE(fromJson(j, q));
    EXPECT_EQ(q.peer, 3);
    EXPECT_EQ(q.characterId, p.characterId);
    EXPECT_EQ(q.moveSpeed, 7.5);

    PlayerInput in;
    in.move = Vec3(1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN());
    in.look = Quat(0.5, 0.5, 0.5, 0.5);
    Json ij = toJson(in);
    EXPECT_EQ(ij["move"][1].string_value(), "inf");
    EXPECT_EQ(ij["look"][0].number_value(), 0.5);
    PlayerInput back;
    ASSERT_TRUE(fromJson(ij, back));
    EXPECT_TRUE(std::isinf(back.move.y()) && std::isnan(back.move.z()));
    EXPECT_EQ(back.look.w(), 0.5);

    Parent par;
    ASSERT_TRUE(fromJson(obj(R"({"entity": "12:3"})"), par));
    EXPECT_EQ(par.entity, (Entity{12, 3}));
    ASSERT_TRUE(fromJson(obj(R"({"entity": {"index": 4, "generation": 1}})"), par));
    EXPECT_EQ(par.entity, (Entity{4, 1}));
    EXPECT_EQ(toJson(par)["entity"]["index"].number_value(), 4);
}

TEST(VisitJson, ErrorsChangeNothing)
{
    Player p;
    p.peer = 1;
    std::string error;
    // Missing keys keep their values
    ASSERT_TRUE(fromJson(obj(R"({"moveSpeed": 2})"), p, &error));
    EXPECT_EQ(p.peer, 1);
    EXPECT_EQ(p.moveSpeed, 2);
    // A later error leaves the earlier field unchanged too (checked before writing)
    EXPECT_FALSE(fromJson(obj(R"({"moveSpeed": 9, "speed": 1})"), p, &error));
    EXPECT_NE(error.find("speed: unknown field"), std::string::npos) << error;
    EXPECT_EQ(p.moveSpeed, 2);
    EXPECT_FALSE(fromJson(obj(R"({"peer": 300})"), p, &error));
    EXPECT_NE(error.find("out of range"), std::string::npos) << error;
    EXPECT_FALSE(fromJson(obj(R"({"peer": 1.5})"), p, &error));
    EXPECT_FALSE(fromJson(obj(R"({"moveSpeed": "fast"})"), p, &error));
    EXPECT_FALSE(fromJson(obj(R"({"characterId": 18014398509481984})"), p, &error));  // 2^54 as a number: inexact
    EXPECT_FALSE(fromJson(obj(R"([1, 2])"), p, &error));
    Parent par;
    EXPECT_FALSE(fromJson(obj(R"({"entity": "12"})"), par, &error));
    EXPECT_FALSE(fromJson(obj(R"({"entity": {"index": 1}})"), par, &error));
    EXPECT_EQ(p.peer, 1);
    EXPECT_EQ(p.moveSpeed, 2);
}

// ---------------------------------------------------------------------------------------------------------------------
// World API

TEST(WorldApi, DescribeStatusSpawnListInspect)
{
    World w(apiSetup());
    WorldApi api(w);
    ApiResult d = api.call("describe_world", Json());
    ASSERT_TRUE(d.ok) << d.error;
    std::set<std::string> components, classes, events;
    for (auto& c : d.value["components"].array_items()) components.insert(c["name"].string_value());
    for (auto& c : d.value["classes"].array_items()) classes.insert(c["name"].string_value());
    for (auto& c : d.value["events"].array_items()) events.insert(c["name"].string_value());
    EXPECT_TRUE(components.count("Health") && components.count("Transform") && components.count("Weather"));
    EXPECT_TRUE(classes.count("Dummy"));
    EXPECT_TRUE(events.count("DamageEvent") && events.count("ActionEvent"));

    // Spawn: a class with initial components, a plain entity
    ApiResult s =
        api.call("spawn", obj(R"({"class": "Dummy", "position": [1, 0, 2], "components": {"Health": {"hp": 40}}})"));
    ASSERT_TRUE(s.ok) << s.error;
    const Entity dummy = entityOf(s.value["entity"]);
    EXPECT_EQ(w.read<Health>(dummy).hp, 40);
    EXPECT_EQ(w.read<Transform>(dummy).position(), Vec3(1, 0, 2));
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(api.call("spawn", obj(R"({"position": [500, 0, 0]})")).ok);
    // A rejected spawn creates nothing
    const uint32_t before = w.registry().aliveCount();
    EXPECT_FALSE(api.call("spawn", obj(R"({"position": [0, 0, 0], "components": {"Health": {"hp": "x"}}})")).ok);
    EXPECT_FALSE(api.call("spawn", obj(R"({"class": "Nope", "position": [0, 0, 0]})")).ok);
    EXPECT_EQ(w.registry().aliveCount(), before);

    ApiResult status = api.call("world_status", Json());
    ASSERT_TRUE(status.ok);
    EXPECT_EQ(status.value["game_objects"].number_value(), 1);

    // list: class filter, component filter, near (grid: after a step), paging
    EXPECT_EQ(api.call("list_entities", obj(R"({"class": "Dummy"})")).value["count"].number_value(), 1);
    EXPECT_EQ(api.call("list_entities", obj(R"({"components": ["Health"]})")).value["count"].number_value(), 1);
    w.step();
    EXPECT_EQ(api.call("list_entities", obj(R"({"near": {"position": [500, 0, 0], "radius": 1}})"))
                  .value["count"]
                  .number_value(),
              5);
    ApiResult page = api.call("list_entities", obj(R"({"limit": 2})"));
    ASSERT_TRUE(page.ok);
    EXPECT_EQ(page.value["count"].number_value(), 2);
    ASSERT_TRUE(page.value["next"].is_number());
    ApiResult page2 = api.call("list_entities", Json(Json::object{{"limit", 2}, {"after", page.value["next"]}}));
    EXPECT_GT(page2.value["entities"][0]["entity"]["index"].number_value(), page.value["next"].number_value());

    ApiResult in = api.call("inspect", Json(Json::object{{"entity", WorldApi::entityJson(dummy)}}));
    ASSERT_TRUE(in.ok) << in.error;
    EXPECT_EQ(in.value["class"].string_value(), "Dummy");
    EXPECT_EQ(in.value["components"]["Health"]["hp"].number_value(), 40);
    EXPECT_EQ(in.value["components"]["Transform"]["position"][2].number_value(), 2);
    EXPECT_TRUE(in.value["flags"]["simulated"].bool_value());

    EXPECT_FALSE(api.call("inspect", obj(R"({"entity": "999:0"})")).ok);
    EXPECT_FALSE(api.call("nope", Json()).ok);
}

TEST(WorldApi, ChangeComponentsTransformsAndHierarchy)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity a             = w.spawn<Dummy>(Vec3::Zero());
    Entity b             = w.create();
    const std::string ea = entityString(a), eb = entityString(b);

    ApiResult r =
        api.call("set_component", obj(R"({"entity": )" + ea + R"(, "component": "Health", "values": {"hp": 5}})"));
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_EQ(w.read<Health>(a).hp, 5);
    // Unknown field: unchanged
    EXPECT_FALSE(api.call("set_component",
                          obj(R"({"entity": )" + ea + R"(, "component": "Health", "values": {"hp": 1, "mana": 2}})"))
                     .ok);
    EXPECT_EQ(w.read<Health>(a).hp, 5);
    // Not on the entity, Behaviour.classId fixed, Persistent not settable
    EXPECT_FALSE(
        api.call("set_component", obj(R"({"entity": )" + eb + R"(, "component": "Health", "values": {"hp": 1}})")).ok);
    EXPECT_FALSE(api.call("set_component",
                          obj(R"({"entity": )" + ea + R"(, "component": "Behaviour", "values": {"classId": 3}})"))
                     .ok);
    ASSERT_TRUE(
        api.call("set_component",
                 obj(R"({"entity": )" + ea + R"(, "component": "Behaviour", "values": {"alwaysTickEveryStep": true}})"))
            .ok);
    EXPECT_TRUE(w.read<Behaviour>(a).alwaysTickEveryStep);

    // Transform through set_component and set_transform (teleport by default)
    ASSERT_TRUE(api.call("set_component", obj(R"({"entity": )" + eb +
                                              R"(, "component": "Transform", "values": {"position": [3, 4, 5]}})"))
                    .ok);
    EXPECT_EQ(w.read<Transform>(b).position(), Vec3(3, 4, 5));
    EXPECT_TRUE(w.read<Transform>(b).wasTeleported());
    EXPECT_FALSE(api.call("set_transform", obj(R"({"entity": )" + eb + R"(, "rotation": [0, 0, 0, 0]})")).ok);
    ASSERT_TRUE(
        api.call("set_transform", obj(R"({"entity": )" + eb + R"(, "rotation": [2, 0, 0, 0], "scale": [2, 2, 2]})"))
            .ok);
    EXPECT_EQ(w.read<Transform>(b).rotation().w(), 1);
    EXPECT_EQ(w.read<Transform>(b).scale(), Vec3(2, 2, 2));

    // add / remove, Parent with the cycle check
    ASSERT_TRUE(
        api.call("add_component", obj(R"({"entity": )" + eb + R"(, "component": "Health", "values": {"maxHp": 7}})"))
            .ok);
    EXPECT_EQ(w.read<Health>(b).maxHp, 7);
    EXPECT_FALSE(api.call("add_component", obj(R"({"entity": )" + eb + R"(, "component": "Health"})")).ok);
    EXPECT_FALSE(api.call("add_component", obj(R"({"entity": )" + eb + R"(, "component": "Behaviour"})")).ok);
    ASSERT_TRUE(api.call("add_component",
                         obj(R"({"entity": )" + eb + R"(, "component": "Parent", "values": {"entity": )" + ea + "}}"))
                    .ok);
    EXPECT_EQ(w.read<Parent>(b).entity, a);
    ApiResult cycle = api.call(
        "add_component", obj(R"({"entity": )" + ea + R"(, "component": "Parent", "values": {"entity": )" + eb + "}}"));
    EXPECT_FALSE(cycle.ok);
    EXPECT_NE(cycle.error.find("cycle"), std::string::npos);
    ASSERT_TRUE(api.call("remove_component", obj(R"({"entity": )" + eb + R"(, "component": "Health"})")).ok);
    EXPECT_FALSE(w.has<Health>(b));
    EXPECT_FALSE(api.call("remove_component", obj(R"({"entity": )" + eb + R"(, "component": "Transform"})")).ok);

    // destroy returns the descendants; onDestroy ran
    ApiResult d = api.call("destroy", obj(R"({"entity": )" + ea + "}"));
    ASSERT_TRUE(d.ok) << d.error;
    EXPECT_EQ(d.value["destroyed"].array_items().size(), 2u);
    EXPECT_FALSE(w.alive(b));
}

TEST(WorldApi, GlobalsEventsActionsAndTime)
{
    World w(apiSetup());
    WorldApi api(w);
    ASSERT_TRUE(api.call("set_global", obj(R"({"component": "Weather", "values": {"rain": 0.5}})")).ok);
    EXPECT_EQ(w.readGlobal<Weather>().rain, 0.5);
    EXPECT_EQ(api.call("get_global", obj(R"({"component": "Weather"})")).value["values"]["rain"].number_value(), 0.5);
    EXPECT_FALSE(api.call("get_global", obj(R"({"component": "Health"})")).ok);
    EXPECT_FALSE(api.call("destroy", Json(Json::object{{"entity", WorldApi::entityJson(w.worldStateEntity())}})).ok);

    // An event through the game's rules: Dummy takes half the damage
    Entity d = w.spawn<Dummy>(Vec3::Zero());
    ASSERT_TRUE(
        api.call("emit_event",
                 Json(Json::object{{"event", "DamageEvent"},
                                   {"values", Json::object{{"target", WorldApi::entityJson(d)}, {"amount", 20}}}}))
            .ok);
    EXPECT_FALSE(api.call("emit_event", obj(R"({"event": "DamageEvent", "values": {"amount": "x"}})")).ok);
    w.step();
    EXPECT_EQ(w.read<Health>(d).hp, 90);

    // Actions: only for players
    Entity p = w.create();
    w.add<Player>(p);
    g_actions.clear();
    EXPECT_FALSE(api.call("queue_action", Json(Json::object{{"player", WorldApi::entityJson(d)}, {"action", 1}})).ok);
    ASSERT_TRUE(api.call("queue_action", Json(Json::object{{"player", WorldApi::entityJson(p)}, {"action", 4}})).ok);
    w.step();
    EXPECT_EQ(g_actions, std::vector<uint32_t>{4});

    // Time: step only while paused
    EXPECT_FALSE(api.call("step", obj(R"({"n": 3})")).ok);
    ASSERT_TRUE(api.call("pause", Json()).ok);
    const uint64_t s0 = w.stepIndex();
    ASSERT_TRUE(api.call("step", obj(R"({"n": 3})")).ok);
    EXPECT_EQ(w.stepIndex(), s0 + 3);
    EXPECT_FALSE(api.call("step", obj(R"({"n": 0})")).ok);
    ASSERT_TRUE(api.call("time_scale", obj(R"({"scale": 2})")).ok);
    EXPECT_EQ(w.timeScale(), 2);
    EXPECT_FALSE(api.call("time_scale", obj(R"({"scale": -1})")).ok);
    ASSERT_TRUE(api.call("resume", Json()).ok);
    EXPECT_FALSE(w.isPaused());
}

TEST(WorldApi, ChangesSinceProfileLogsAndApply)
{
    World w(apiSetup());
    WorldApi api(w);
    // include_moved starts the move tracking
    ApiResult c0 = api.call("changes_since", obj(R"({"cursor": 0, "include_moved": true})"));
    ASSERT_TRUE(c0.ok) << c0.error;
    const Json cursor = c0.value["cursor"];
    Entity a          = w.create();
    w.add<Health>(a);
    w.step();
    w.get<Transform>(a).setPosition(Vec3(1, 0, 0));
    w.step();
    Entity b = w.create();
    w.destroy(a);
    ApiResult c1 = api.call("changes_since", Json(Json::object{{"cursor", cursor}, {"include_moved", true}}));
    ASSERT_TRUE(c1.ok) << c1.error;
    std::vector<std::string> kinds;
    for (auto& c : c1.value["changes"].array_items()) kinds.push_back(c["kind"].string_value());
    // The components of a destroyed entity are folded into "destroyed"
    EXPECT_EQ(kinds, (std::vector<std::string>{"created", "added", "created", "destroyed"}));
    // Moves are coalesced per entity: only alive ones (a is gone, b never moved)
    EXPECT_TRUE(c1.value["moved"].array_items().empty());
    w.get<Transform>(b).setPosition(Vec3(2, 0, 0));
    w.step();
    w.step();
    ApiResult c2 =
        api.call("changes_since", Json(Json::object{{"cursor", c1.value["cursor"]}, {"include_moved", true}}));
    ASSERT_EQ(c2.value["moved"].array_items().size(), 1u);
    EXPECT_EQ(entityOf(c2.value["moved"][0]), b);
    // A cursor from another session is an error (it would skip every change)
    EXPECT_FALSE(api.call("changes_since", obj(R"({"cursor": "100000@5"})")).ok);
    EXPECT_FALSE(api.call("changes_since", obj(R"({"cursor": "garbage"})")).ok);
    // Nothing new after the returned cursor
    EXPECT_TRUE(api.call("changes_since", Json(Json::object{{"cursor", c2.value["cursor"]}}))
                    .value["changes"]
                    .array_items()
                    .empty());

    ApiResult prof = api.call("get_profile", obj(R"({"steps": 2})"));
    ASSERT_TRUE(prof.ok);
    EXPECT_EQ(prof.value["samples"].number_value(), 2);
    bool hasBehaviour = false;
    for (auto& s : prof.value["scopes"].array_items())
        hasBehaviour = hasBehaviour || s["scope"].string_value() == "step/phase:behaviour/system:behaviour";
    EXPECT_TRUE(hasBehaviour);

    // apply: in order, stops at the first failure
    Entity target        = w.create();
    const std::string eb = entityString(target);
    ApiResult batch      = api.call("apply", obj(R"({"operations": [
        {"op": "add_component", "entity": )" +
                                                 eb + R"(, "component": "Health"},
        {"op": "set_component", "entity": )" +
                                                 eb + R"(, "component": "Health", "values": {"hp": 3}},
        {"op": "set_component", "entity": "999:0", "component": "Health", "values": {"hp": 4}},
        {"op": "set_component", "entity": )" +
                                                 eb + R"(, "component": "Health", "values": {"hp": 5}}]})"));
    EXPECT_FALSE(batch.ok);
    EXPECT_NE(batch.error.find("operation 2"), std::string::npos) << batch.error;
    EXPECT_EQ(batch.value["applied"].number_value(), 2);
    EXPECT_EQ(w.read<Health>(target).hp, 3);
    EXPECT_FALSE(api.call("apply", obj(R"({"operations": [{"op": "apply"}]})")).ok);

    // The edits are logged under "api"
    ApiResult l = api.call("logs", obj(R"({"category": "api"})"));
    ASSERT_TRUE(l.ok);
    EXPECT_FALSE(l.value["lines"].array_items().empty());
}

TEST(WorldApi, ReadOnlyAndNotInsideAStep)
{
    World w(apiSetup());
    WorldApi readOnly(w, false);
    EXPECT_TRUE(readOnly.call("world_status", Json()).ok);
    EXPECT_FALSE(readOnly.call("spawn", obj(R"({"position": [0, 0, 0]})")).ok);
    for (auto& op : readOnly.operations()) EXPECT_FALSE(WorldApi::isEdit(op)) << op;

    static WorldApi* inStepApi = nullptr;
    static bool refused        = false;
    WorldSetup s               = apiSetup();
    s.systems                  = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "api_in_step",
              [](World&, double) { refused = inStepApi && !inStepApi->call("world_status", Json()).ok; });
    };
    World w2(s);
    WorldApi api2(w2);
    inStepApi = &api2;
    w2.step();
    inStepApi = nullptr;
    EXPECT_TRUE(refused);
}

// ---------------------------------------------------------------------------------------------------------------------
// MCP tools

namespace
{
Json callTool(McpServer& server, const std::string& name, const Json& args)
{
    return server.handleMessage(McpRequest("tools/call", Json::object{{"name", name}, {"arguments", args}}));
}
Json toolJson(const Json& response)
{
    std::string err;
    Json j = Json::parse(response["result"]["content"][0]["text"].string_value(), err);
    return j;
}
}  // namespace

TEST(WorldTools, EveryToolThroughHandleMessage)
{
    World w(apiSetup());
    McpServer server("test");
    auto api  = registerWorldTools(server, w);
    Json list = server.handleMessage(McpRequest("tools/list", Json::object{}));
    std::set<std::string> names;
    for (auto& t : list["result"]["tools"].array_items()) names.insert(t["name"].string_value());
    for (const std::string& op : api->operations()) EXPECT_TRUE(names.count(op)) << op;
    EXPECT_TRUE(names.count("measure_performance"));
    EXPECT_EQ(names.size(), api->operations().size() + 1);

    Poller poller(server);
    // Reading
    Json status = callTool(server, "world_status", Json::object{});
    ASSERT_FALSE(status["result"]["isError"].bool_value());
    EXPECT_EQ(toolJson(status)["step"].number_value(), 0);
    EXPECT_FALSE(callTool(server, "describe_world", Json::object{})["result"]["isError"].bool_value());
    // Editing and errors as isError results
    Json spawned = callTool(server, "spawn", obj(R"({"class": "Dummy", "position": [0, 0, 0]})"));
    ASSERT_FALSE(spawned["result"]["isError"].bool_value()) << spawned.dump();
    const Entity d = entityOf(toolJson(spawned)["entity"]);
    Json bad = callTool(server, "set_component", obj(R"({"entity": "999:0", "component": "Health", "values": {}})"));
    EXPECT_TRUE(bad["result"]["isError"].bool_value());
    Json set = callTool(
        server, "set_component",
        Json::object{
            {"entity", WorldApi::entityJson(d)}, {"component", "Health"}, {"values", Json::object{{"hp", 12}}}});
    ASSERT_FALSE(set["result"]["isError"].bool_value());
    EXPECT_EQ(toolJson(set)["values"]["hp"].number_value(), 12);
    EXPECT_EQ(toolJson(set)["previous"]["hp"].number_value(), 100);  // for an undo
    // Paused: measure_performance runs exactly n steps
    callTool(server, "pause", Json::object{});
    Json m = callTool(server, "measure_performance", obj(R"({"steps": 5})"));
    ASSERT_FALSE(m["result"]["isError"].bool_value());
    EXPECT_EQ(toolJson(m)["samples"].number_value(), 5);
    EXPECT_EQ(w.stepIndex(), 5u);
}

TEST(WorldTools, MeasurePerformanceWhileRunningAndReadOnly)
{
    World w(apiSetup());
    McpServer server("test");
    auto api = registerWorldTools(server, w);
    {
        // The poller thread is the app's main loop: poll, then a step
        Poller poller(server, [&]() { w.step(); });
        Json m = callTool(server, "measure_performance", obj(R"({"steps": 10})"));
        ASSERT_FALSE(m["result"]["isError"].bool_value()) << m.dump();
        EXPECT_EQ(toolJson(m)["samples"].number_value(), 10);
    }
    McpServer observer("observer");
    auto readOnly = registerWorldTools(observer, w, WorldToolsOptions{false});
    Json list     = observer.handleMessage(McpRequest("tools/list", Json::object{}));
    std::set<std::string> names;
    for (auto& t : list["result"]["tools"].array_items()) names.insert(t["name"].string_value());
    EXPECT_TRUE(names.count("inspect"));
    EXPECT_FALSE(names.count("set_component"));
    EXPECT_FALSE(names.count("spawn"));
    // The API is released: the tools answer with an error
    api.reset();
    Poller poller(server);
    EXPECT_TRUE(callTool(server, "world_status", Json::object{})["result"]["isError"].bool_value());
}

// ---------------------------------------------------------------------------------------------------------------------
// Review regressions (S10 round 1)

// Regression: 2^53 + 1 was written as the number 2^53 (the bound was compared in double), and the literal
// 9007199254740993 was accepted (parsed as 2^53)
TEST(VisitJson, IntegersAtTheDoubleBoundary)
{
    Player p;
    p.characterId = (uint64_t(1) << 53) + 1;
    Json j        = toJson(p);
    ASSERT_TRUE(j["characterId"].is_string());
    Player q;
    ASSERT_TRUE(fromJson(j, q));
    EXPECT_EQ(q.characterId, p.characterId);
    p.characterId = (uint64_t(1) << 53) - 1;
    EXPECT_TRUE(toJson(p)["characterId"].is_number());
    std::string error;
    EXPECT_FALSE(fromJson(obj(R"({"characterId": 9007199254740993})"), q, &error));
    EXPECT_FALSE(fromJson(obj(R"({"characterId": -1})"), q, &error));
    // An unknown field names the valid ones
    EXPECT_FALSE(fromJson(obj(R"({"speed": 1})"), q, &error));
    EXPECT_NE(error.find("fields: peer, characterId, moveSpeed"), std::string::npos) << error;
}

TEST(WorldApi, StrictArgumentsAndGuards)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity a = w.create();
    Entity c = w.create();
    w.add<Parent>(c, Parent{a});
    // Unknown arguments, wrong types
    ApiResult typo = api.call(
        "set_transform", Json(Json::object{{"entity", WorldApi::entityJson(a)}, {"postion", Json::array{1, 2, 3}}}));
    EXPECT_FALSE(typo.ok);
    EXPECT_NE(typo.error.find("unknown argument \"postion\""), std::string::npos) << typo.error;
    EXPECT_FALSE(api.call("list_entities", obj(R"({"after": "x"})")).ok);
    EXPECT_FALSE(api.call("list_entities", obj(R"({"after": 1e20})")).ok);
    EXPECT_FALSE(api.call("list_entities", obj(R"({"limit": 2.5})")).ok);
    EXPECT_FALSE(
        api.call("set_transform", Json(Json::object{{"entity", WorldApi::entityJson(a)}, {"teleport", "yes"}})).ok);
    // A child follows its parent: set_transform is refused
    ApiResult child = api.call(
        "set_transform", Json(Json::object{{"entity", WorldApi::entityJson(c)}, {"position", Json::array{1, 2, 3}}}));
    EXPECT_FALSE(child.ok);
    EXPECT_NE(child.error.find("LocalTransform"), std::string::npos);
    // World state: no Parent, no destroy (also not through a cascade)
    EXPECT_FALSE(
        api.call("add_component", Json(Json::object{{"entity", WorldApi::entityJson(w.worldStateEntity())},
                                                    {"component", "Parent"},
                                                    {"values", Json::object{{"entity", WorldApi::entityJson(a)}}}}))
            .ok);
    EXPECT_FALSE(api.call("remove_component", Json(Json::object{{"entity", WorldApi::entityJson(w.worldStateEntity())},
                                                                {"component", "Weather"}}))
                     .ok);
    EXPECT_TRUE(w.alive(w.worldStateEntity()));
    // Behaviour: tick flags settable, the interval derived again, classId fixed, no negative times
    Entity d = w.spawn<Dummy>(Vec3(9000, 0, 0));
    EXPECT_EQ(w.read<Behaviour>(d).tickInterval, 0.5);
    // tickInterval is derived: an error that points to alwaysTickEveryStep
    ApiResult derived = api.call("set_component", Json(Json::object{{"entity", WorldApi::entityJson(d)},
                                                                    {"component", "Behaviour"},
                                                                    {"values", Json::object{{"tickInterval", 0}}}}));
    EXPECT_FALSE(derived.ok);
    EXPECT_NE(derived.error.find("alwaysTickEveryStep"), std::string::npos);
    ASSERT_TRUE(api.call("set_component", Json(Json::object{{"entity", WorldApi::entityJson(d)},
                                                            {"component", "Behaviour"},
                                                            {"values", Json::object{{"alwaysTickEveryStep", true}}}}))
                    .ok);
    // The interval stays derived from the rings (the flag is read by the behaviour system): clearing it takes effect
    EXPECT_EQ(w.read<Behaviour>(d).tickInterval, 0.5);
    w.get<Behaviour>(d).alwaysTickEveryStep = false;
    EXPECT_EQ(w.read<Behaviour>(d).tickInterval, 0.5);
    EXPECT_FALSE(api.call("set_component", Json(Json::object{{"entity", WorldApi::entityJson(d)},
                                                             {"component", "Behaviour"},
                                                             {"values", Json::object{{"sinceTick", -1}}}}))
                     .ok);
    EXPECT_FALSE(api.call("set_component", Json(Json::object{{"entity", WorldApi::entityJson(d)},
                                                             {"component", "Behaviour"},
                                                             {"values", Json::object{{"classId", 7}}}}))
                     .ok);
    // apply in read-only mode: reads only
    WorldApi readOnly(w, false);
    EXPECT_TRUE(
        readOnly.call("apply", obj(R"({"operations": [{"op": "world_status"}, {"op": "describe_world"}]})")).ok);
    EXPECT_FALSE(readOnly.call("apply", obj(R"({"operations": [{"op": "pause"}]})")).ok);
}

// Regression: the API kept a reference to a destroyed world
TEST(WorldApi, DetachesWhenTheWorldIsDestroyed)
{
    auto w = std::make_unique<World>(apiSetup());
    WorldApi api(*w);
    EXPECT_TRUE(api.call("world_status", Json()).ok);
    w.reset();
    EXPECT_FALSE(api.call("world_status", Json()).ok);
}

// ---------------------------------------------------------------------------------------------------------------------
// Profiler

TEST(Profiler, StatisticsRingTreeAndCounters)
{
    Profiler p(4);
    const int step  = p.scope("step");
    const int child = p.scope("step/phase:a");
    const int c     = p.counter("n");
    for (int i = 1; i <= 6; ++i)
    {
        p.add(step, 10.0 * i);
        p.add(child, 2.5 * i);
        p.add(child, 2.5 * i);  // adds up within a sample
        p.setCounter(c, i);
        p.endSample();
    }
    // The ring keeps the last 4 samples: 30, 40, 50, 60
    Profiler::Stats s = p.scopeStats(step);
    EXPECT_EQ(s.samples, 4);
    EXPECT_EQ(s.min, 30);
    EXPECT_EQ(s.max, 60);
    EXPECT_EQ(s.avg, 45);
    EXPECT_EQ(s.p50, 40);  // nearest rank
    EXPECT_EQ(p.scopeStats(step, 2).avg, 55);
    EXPECT_EQ(p.counterStats(c).max, 6);
    Json r = p.report();
    ASSERT_EQ(r["scopes"].array_items().size(), 2u);
    EXPECT_NEAR(r["scopes"][1]["share_of_parent"].number_value(), 0.5, 1e-12);
    // A discarded sample leaves no trace
    p.add(step, 1000);
    p.discardSample();
    p.endSample();
    EXPECT_EQ(p.scopeStats(step, 1).max, 0);
    p.reset();
    EXPECT_EQ(p.scopeStats(step).samples, 0);
}

namespace
{
bool g_slowThrow = false;
void sysSlowThrow(World&, double)
{
    if (!g_slowThrow) return;
    g_slowThrow = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    throw std::runtime_error("aborted");
}
}  // namespace

// Regression: an aborted step's time was added to the next sample (the step timer ended after the discard)
TEST(Profiler, AnAbortedStepIsNotCounted)
{
    WorldSetup s = apiSetup();
    s.systems    = [](SystemList& l) { l.add(Phase::Gameplay, "slow_throw", sysSlowThrow); };
    World w(s);
    g_slowThrow = true;
    EXPECT_THROW(w.step(), std::runtime_error);
    w.step();
    const int scope = w.profiler().scope("step");
    EXPECT_LT(w.profiler().scopeStats(scope, 1).max, 25.0);  // the aborted step slept 30 ms
    EXPECT_EQ(w.profiler().sampleCount(), 1u);
}

// R-MCP-1: every tool through handleMessage (the error paths are covered by the API tests)
TEST(WorldTools, EveryToolAnswers)
{
    World w(apiSetup());
    McpServer server("test");
    WorldToolsOptions options;
    options.snapshotDirectory = ::testing::TempDir();
    auto api                  = registerWorldTools(server, w, options);
    Poller poller(server);
    auto call = [&](const std::string& name, const Json& args)
    {
        Json r = callTool(server, name, args);
        EXPECT_FALSE(r["result"]["isError"].bool_value()) << name << ": " << r.dump();
        return toolJson(r);
    };
    std::set<std::string> called;
    auto track = [&](const std::string& name, const Json& args)
    {
        called.insert(name);
        return call(name, args);
    };
    track("describe_world", Json::object{});
    track("world_status", Json::object{});
    const Entity d =
        entityOf(track("spawn", obj(R"({"class": "Dummy", "position": [12345.678, 0, 0.000123456789]})"))["entity"]);
    const Entity p =
        entityOf(track("spawn", obj(R"({"position": [0, 0, 0], "components": {"Player": {}}})"))["entity"]);
    const Json de = WorldApi::entityJson(d);
    // Exact numbers in the results (not shortened to 6 digits)
    Json in = track("inspect", Json::object{{"entity", de}});
    EXPECT_EQ(in["components"]["Transform"]["position"][0].number_value(), 12345.678);
    EXPECT_EQ(in["components"]["Transform"]["position"][2].number_value(), 0.000123456789);
    track("list_entities", obj(R"({"class": "Dummy"})"));
    track("set_component", Json::object{{"entity", de}, {"component", "Health"}, {"values", Json::object{{"hp", 3}}}});
    track("add_component", Json::object{{"entity", de}, {"component", "PlayerInput"}});
    track("remove_component", Json::object{{"entity", de}, {"component", "PlayerInput"}});
    track("set_transform", Json::object{{"entity", de}, {"position", Json::array{1, 2, 3}}});
    track("get_global", obj(R"({"component": "Weather"})"));
    track("set_global", obj(R"({"component": "Weather", "values": {"rain": 1}})"));
    track("emit_event",
          Json::object{{"event", "DamageEvent"}, {"values", Json::object{{"target", de}, {"amount", 2}}}});
    track("queue_action", Json::object{{"player", WorldApi::entityJson(p)}, {"action", 7}});
    track("changes_since", obj(R"({"include_moved": true})"));
    track("pause", Json::object{});
    track("step", obj(R"({"n": 2})"));
    track("time_scale", obj(R"({"scale": 1})"));
    track("resume", Json::object{});
    track("get_profile", obj(R"({"scopes": ["step/phase:behaviour"]})"));
    track("reset_profile", Json::object{});
    track("logs", obj(R"({"category": "api"})"));
    track("apply", obj(R"({"operations": [{"op": "world_status"}]})"));
    track("save_snapshot", obj(R"({"name": "before"})"));
    track("list_snapshots", Json::object{});
    track("diff_snapshot", obj(R"({"name": "before"})"));
    track("export_snapshot", obj(R"({"name": "before", "file": "saiga_tools_before.sgws", "overwrite": true})"));
    track("import_snapshot", obj(R"({"name": "again", "file": "saiga_tools_before.sgws"})"));
    track("load_snapshot", obj(R"({"name": "again"})"));
    track("delete_snapshot", obj(R"({"name": "again"})"));
    track("get_data", obj(R"({"type": "CreatureStats", "name": "wolf"})"));
    track("reload_data", Json::object{});
    track("destroy", Json::object{{"entity", de}});
    track("pause", Json::object{});
    track("measure_performance", obj(R"({"steps": 3})"));
    std::remove((::testing::TempDir() + "/saiga_tools_before.sgws").c_str());
    for (const std::string& op : api->operations()) EXPECT_TRUE(called.count(op)) << op << " has no test call";
    // At most the profile history
    EXPECT_TRUE(callTool(server, "measure_performance", obj(R"({"steps": 601})"))["result"]["isError"].bool_value());
}

// ---------------------------------------------------------------------------------------------------------------------
// Review regressions (S10 round 2)

// Every tool schema lists exactly the arguments of its operation (a client that validates schemas must not reject
// arguments the API accepts, and an agent reading the schema must see every option)
TEST(WorldTools, SchemasMatchTheOperations)
{
    World w(apiSetup());
    McpServer server("test");
    auto api    = registerWorldTools(server, w);
    Json list   = server.handleMessage(McpRequest("tools/list", Json::object{}));
    int checked = 0;
    for (auto& t : list["result"]["tools"].array_items())
    {
        const std::string name = t["name"].string_value();
        EXPECT_EQ(t["inputSchema"]["additionalProperties"], Json(false)) << name;
        if (name == "measure_performance") continue;
        std::set<std::string> schema, op;
        for (auto& kv : t["inputSchema"]["properties"].object_items()) schema.insert(kv.first);
        for (auto& a : WorldApi::arguments(name)) op.insert(a);
        EXPECT_EQ(schema, op) << name;
        checked++;
    }
    EXPECT_EQ(checked, int(api->operations().size()));
    // Read-only: apply stays (batches of reads)
    McpServer observer("observer");
    auto ro = registerWorldTools(observer, w, WorldToolsOptions{false});
    Poller poller(observer);
    Json r = callTool(observer, "apply", obj(R"({"operations": [{"op": "world_status"}]})"));
    EXPECT_FALSE(r["result"]["isError"].bool_value()) << r.dump();
}

// Regression: with more than a page of moved entities, moved_more returned the same page forever
TEST(WorldApi, MovedEntitiesArePagedAndCoalesced)
{
    World w(apiSetup());
    WorldApi api(w);
    std::vector<Entity> movers;
    for (int i = 0; i < 25; ++i) movers.push_back(w.create());
    ApiResult c = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true, "limit": 10})"));
    ASSERT_TRUE(c.ok) << c.error;
    // Every mover moves in three steps: reported once each
    for (int s = 0; s < 3; ++s)
    {
        for (Entity e : movers) w.get<Transform>(e).setPosition(Vec3(double(s), 0, 0));
        w.step();
    }
    std::set<uint32_t> seen;
    Json cursor = c.value["cursor"];
    for (int page = 0; page < 5; ++page)
    {
        ApiResult r =
            api.call("changes_since", Json(Json::object{{"cursor", cursor}, {"include_moved", true}, {"limit", 10}}));
        ASSERT_TRUE(r.ok) << r.error;
        for (auto& m : r.value["moved"].array_items())
        {
            EXPECT_TRUE(seen.insert(uint32_t(m["index"].number_value())).second) << "reported twice";
        }
        cursor = r.value["cursor"];
        if (!r.value["moved_more"].bool_value()) break;
    }
    EXPECT_EQ(seen.size(), movers.size());
    // A cursor of another API (session) is refused
    WorldApi other(w);
    EXPECT_FALSE(other.call("changes_since", Json(Json::object{{"cursor", cursor}})).ok);
}

// Regression: agents could write nan and inf (and a zero quaternion) into components through the generic path
TEST(WorldApi, WritesMustBeFinite)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity e      = w.create();
    const Json ej = WorldApi::entityJson(e);
    EXPECT_FALSE(
        api.call("add_component",
                 Json(Json::object{{"entity", ej}, {"component", "Health"}, {"values", Json::object{{"hp", "nan"}}}}))
            .ok);
    EXPECT_FALSE(
        api.call("add_component",
                 obj(R"({"entity": "1:0", "component": "LocalTransform", "values": {"rotation": [0, 0, 0, 0]}})"))
            .ok);
    ASSERT_TRUE(api.call("add_component", Json(Json::object{{"entity", ej}, {"component", "Health"}})).ok);
    EXPECT_FALSE(
        api.call("set_component",
                 Json(Json::object{{"entity", ej}, {"component", "Health"}, {"values", obj(R"({"hp": 1e400})")}}))
            .ok);
    EXPECT_EQ(w.read<Health>(e).hp, 100);
    EXPECT_FALSE(api.call("emit_event", obj(R"({"event": "DamageEvent", "values": {"amount": "inf"}})")).ok);
}

namespace
{
enum class StrictClass : uint16_t
{
    Turret = kFirstGameClass + 1,
};
// Adds Health in onSpawn without checking (an agent may pass Health as an initial component)
class Turret final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(StrictClass::Turret);
    static constexpr const char* name = "Turret";
    void onSpawn() override { add<Health>(Health{10, 10}); }
};
}  // namespace

// Regression: an agent's spawn with an initial component that the class adds itself asserted (double add)
TEST(WorldApi, SpawnMergesInitialComponentsTheObjectAdded)
{
    WorldSetup s = apiSetup();
    s.classes    = [](ClassList& c)
    {
        c.add<Dummy>();
        c.add<Turret>();
    };
    World w(s);
    WorldApi api(w);
    ApiResult r = api.call("spawn", obj(R"({"class": "Turret", "position": [0, 0, 0], "rotation": [0, 0, 1, 0],
                                           "components": {"Health": {"hp": 5}}})"));
    ASSERT_TRUE(r.ok) << r.error;
    const Entity t = entityOf(r.value["entity"]);
    EXPECT_EQ(w.read<Health>(t).hp, 5);
    EXPECT_EQ(w.read<Health>(t).maxHp, 10);
    EXPECT_EQ(w.read<Transform>(t).rotation().y(), 1);
    // list_entities with values and distance
    w.step();
    ApiResult l =
        api.call("list_entities", obj(R"({"near": {"position": [0, 0, 0], "radius": 5}, "include": ["Health"]})"));
    ASSERT_TRUE(l.ok) << l.error;
    ASSERT_EQ(l.value["count"].number_value(), 1);
    EXPECT_EQ(l.value["entities"][0]["values"]["Health"]["hp"].number_value(), 5);
    EXPECT_EQ(l.value["entities"][0]["distance"].number_value(), 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// Review regressions (S10 round 3)

namespace
{
enum class Round3Class : uint16_t
{
    Spinner = kFirstGameClass + 2,
    Killer,
};
// Sets its own rotation and scale in onSpawn
class Spinner final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(Round3Class::Spinner);
    static constexpr const char* name = "Spinner";
    void onSpawn() override
    {
        transformWrite().setRotation(Quat(0, 0, 1, 0));
        transformWrite().setScale(Vec3(2, 2, 2));
    }
};
Entity g_victim;
// Destroys an entity in onSpawn (the parent an agent passed as an initial component)
class Killer final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(Round3Class::Killer);
    static constexpr const char* name = "Killer";
    void onSpawn() override { world().destroy(g_victim); }
};
}  // namespace

// Regression: spawn reset a class's own rotation and scale, and an initial Parent destroyed by onSpawn asserted
TEST(WorldApi, SpawnKeepsTheClassTransformAndReportsSkippedComponents)
{
    WorldSetup s = apiSetup();
    s.classes    = [](ClassList& c)
    {
        c.add<Dummy>();
        c.add<Turret>();  // the class IDs are contiguous
        c.add<Spinner>();
        c.add<Killer>();
    };
    World w(s);
    WorldApi api(w);
    ApiResult r = api.call("spawn", obj(R"({"class": "Spinner", "position": [0, 0, 0]})"));
    ASSERT_TRUE(r.ok) << r.error;
    const Entity sp = entityOf(r.value["entity"]);
    EXPECT_EQ(w.read<Transform>(sp).scale(), Vec3(2, 2, 2));
    EXPECT_EQ(w.read<Transform>(sp).rotation().y(), 1);
    g_victim = w.create();
    r        = api.call(
        "spawn",
        Json(Json::object{
                   {"class", "Killer"},
                   {"position", Json::array{0, 0, 0}},
                   {"components", Json::object{{"Parent", Json::object{{"entity", WorldApi::entityJson(g_victim)}}}}}}));
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_EQ(r.value["skipped"][0]["component"].string_value(), "Parent");
    EXPECT_FALSE(w.has<Parent>(entityOf(r.value["entity"])));
}

// Regression: a unit quaternion read and written back changed by 1 ulp (normalized again)
TEST(VisitJson, UnitQuaternionsRoundTripExactly)
{
    Rng rng(3);
    for (int i = 0; i < 1000; ++i)
    {
        LocalTransform t;
        Quat q(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
        q.normalize();
        t.rotation = q;
        LocalTransform back;
        ASSERT_TRUE(fromJson(toJson(t), back, nullptr, true));
        EXPECT_EQ(back.rotation.w(), t.rotation.w());
        EXPECT_EQ(back.rotation.x(), t.rotation.x());
    }
    // Not a unit quaternion: normalized (finiteOnly); zero: refused
    LocalTransform t;
    ASSERT_TRUE(fromJson(obj(R"({"rotation": [2, 0, 0, 0]})"), t, nullptr, true));
    EXPECT_EQ(t.rotation.w(), 1);
    EXPECT_FALSE(fromJson(obj(R"({"rotation": [0, 0, 0, 0]})"), t, nullptr, true));
}

// Regression: with more structural changes than a page, every page repeated the whole moved list
TEST(WorldApi, TheMovedListIsNotRepeatedWhileChangesArePaged)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity mover = w.create();
    ApiResult c  = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})"));
    w.get<Transform>(mover).setPosition(Vec3(1, 0, 0));
    for (int i = 0; i < 30; ++i) w.create();  // 30 structural changes
    w.step();
    size_t movedReports = 0;
    Json cursor         = c.value["cursor"];
    for (int page = 0; page < 10; ++page)
    {
        ApiResult r =
            api.call("changes_since", Json(Json::object{{"cursor", cursor}, {"include_moved", true}, {"limit", 10}}));
        ASSERT_TRUE(r.ok) << r.error;
        movedReports += r.value["moved"].array_items().size();
        cursor = r.value["cursor"];
        if (!r.value["more"].bool_value() && !r.value["moved_more"].bool_value()) break;
    }
    EXPECT_EQ(movedReports, 1u);
}

// ---------------------------------------------------------------------------------------------------------------------
// CodeRabbit regressions (PR #52)

// near selects by the grid position of the last step: the result reports it and the distance from it
TEST(WorldApi, NearResultsUseTheGridPosition)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity e = w.create();
    w.get<Transform>(e).setPosition(Vec3(1, 0, 0));
    w.step();
    w.get<Transform>(e).setPosition(Vec3(100, 0, 0));
    ApiResult r = api.call("list_entities", obj(R"({"near": {"position": [0, 0, 0], "radius": 2}})"));
    ASSERT_EQ(r.value["count"].number_value(), 1);
    EXPECT_EQ(r.value["entities"][0]["distance"].number_value(), 1);
    EXPECT_EQ(r.value["entities"][0]["grid_position"][0].number_value(), 1);
    EXPECT_EQ(r.value["entities"][0]["position"][0].number_value(), 100);
}

// A move recorded for an entity whose slot is reused is not reported for the new entity; moves made while a window is
// paged are reported in the next window
TEST(WorldApi, MoveWindowsKeepLateMovesAndHandles)
{
    World w(apiSetup());
    WorldApi api(w);
    std::vector<Entity> e;
    for (int i = 0; i < 4; ++i) e.push_back(w.create());
    Json cursor = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})")).value["cursor"];
    // Step 1: e[1] and e[3] move; e[2] moves and is destroyed, its slot reused
    w.get<Transform>(e[1]).setPosition(Vec3(1, 0, 0));
    w.get<Transform>(e[2]).setPosition(Vec3(1, 0, 0));
    w.get<Transform>(e[3]).setPosition(Vec3(1, 0, 0));
    w.step();
    w.destroy(e[2]);
    Entity reused = w.create();
    ASSERT_EQ(reused.index, e[2].index);
    // Page one of the window
    ApiResult p1 =
        api.call("changes_since", Json(Json::object{{"cursor", cursor}, {"include_moved", true}, {"limit", 1}}));
    ASSERT_TRUE(p1.ok) << p1.error;
    ASSERT_TRUE(p1.value["moved_more"].bool_value());
    // Step 2 before page two: e[0] (behind the page index) moves
    w.get<Transform>(e[0]).setPosition(Vec3(2, 0, 0));
    w.step();
    std::set<uint32_t> reported;
    for (auto& m : p1.value["moved"].array_items()) reported.insert(uint32_t(m["index"].number_value()));
    Json c = p1.value["cursor"];
    for (int page = 0; page < 6; ++page)
    {
        ApiResult r =
            api.call("changes_since", Json(Json::object{{"cursor", c}, {"include_moved", true}, {"limit", 1}}));
        ASSERT_TRUE(r.ok) << r.error;
        for (auto& m : r.value["moved"].array_items())
        {
            EXPECT_EQ(uint32_t(m["generation"].number_value()),
                      w.registry().aliveAt(uint32_t(m["index"].number_value())).generation);
            reported.insert(uint32_t(m["index"].number_value()));
        }
        c = r.value["cursor"];
    }
    EXPECT_TRUE(reported.count(e[0].index)) << "a move behind the page index was lost";
    EXPECT_TRUE(reported.count(e[1].index) && reported.count(e[3].index));
    EXPECT_FALSE(reported.count(reused.index)) << "the reused slot was reported as moved";
}

// The previous values of set_component are accepted by set_component (an undo)
TEST(WorldApi, PreviousValuesAreAnUndo)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity d      = w.spawn<Dummy>(Vec3(1, 2, 3));
    const Json de = WorldApi::entityJson(d);
    ApiResult t   = api.call(
        "set_component",
        Json(Json::object{{"entity", de}, {"component", "Transform"}, {"values", obj(R"({"position": [5, 5, 5]})")}}));
    ASSERT_TRUE(t.ok) << t.error;
    ApiResult undo =
        api.call("set_component",
                 Json(Json::object{{"entity", de}, {"component", "Transform"}, {"values", t.value["previous"]}}));
    ASSERT_TRUE(undo.ok) << undo.error;
    EXPECT_EQ(w.read<Transform>(d).position(), Vec3(1, 2, 3));
    ApiResult b = api.call("set_component", Json(Json::object{{"entity", de},
                                                              {"component", "Behaviour"},
                                                              {"values", obj(R"({"alwaysTickEveryStep": true})")}}));
    ASSERT_TRUE(b.ok) << b.error;
    EXPECT_TRUE(
        api.call("set_component",
                 Json(Json::object{{"entity", de}, {"component", "Behaviour"}, {"values", b.value["previous"]}}))
            .ok);
    EXPECT_FALSE(w.read<Behaviour>(d).alwaysTickEveryStep);
}

namespace
{
enum class ThrowClass : uint16_t
{
    Faulty = kFirstGameClass + 1,
};
class Faulty final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ThrowClass::Faulty);
    static constexpr const char* name = "Faulty";
    void onSpawn() override { throw std::runtime_error("broken onSpawn"); }
};
}  // namespace

// A throwing hook inside apply: an indexed failure with the results of the earlier operations
TEST(WorldApi, AThrowingHookInApplyKeepsTheEarlierResults)
{
    WorldSetup s = apiSetup();
    s.classes    = [](ClassList& c)
    {
        c.add<Dummy>();
        c.add<Faulty>();
    };
    World w(s);
    WorldApi api(w);
    ApiResult r = api.call("apply", obj(R"({"operations": [{"op": "spawn", "position": [0, 0, 0]},
                                                         {"op": "spawn", "class": "Faulty", "position": [0, 0, 0]}]})"));
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("operation 1"), std::string::npos) << r.error;
    EXPECT_NE(r.error.find("broken onSpawn"), std::string::npos) << r.error;
    ASSERT_EQ(r.value["applied"].number_value(), 1);
    EXPECT_TRUE(w.alive(entityOf(r.value["results"][0]["entity"])));
}

// A huge quaternion is normalized without overflow (its squared length was inf: stored as zero)
TEST(VisitJson, HugeQuaternionsAreNormalized)
{
    LocalTransform t;
    ASSERT_TRUE(fromJson(obj(R"({"rotation": [1e200, 0, 0, 0]})"), t, nullptr, true));
    EXPECT_EQ(t.rotation.w(), 1);
}

// Tool numbers do not depend on the process locale (a comma separator split numbers)
TEST(WorldTools, NumbersIgnoreTheLocale)
{
    World w(apiSetup());
    McpServer server("test");
    auto api = registerWorldTools(server, w);
    Entity e = w.create();
    w.get<Transform>(e).setPosition(Vec3(1.5, 0, 0));
    const char* old         = std::setlocale(LC_NUMERIC, nullptr);
    const std::string saved = old ? old : "C";
    const bool german       = std::setlocale(LC_NUMERIC, "de_DE.UTF-8") != nullptr;
    Poller poller(server);
    Json r = callTool(server, "inspect", Json::object{{"entity", WorldApi::entityJson(e)}});
    std::setlocale(LC_NUMERIC, saved.c_str());
    EXPECT_EQ(toolJson(r)["components"]["Transform"]["position"][0].number_value(), 1.5) << (german ? "de_DE" : "C");
}

// A running measurement answers at its deadline although the world stopped stepping, and fails on a profile reset
TEST(WorldTools, MeasurementDeadlineWithoutStepsAndReset)
{
    World w(apiSetup());
    McpServer server("test");
    server.toolTimeoutSeconds = 1;
    auto api                  = registerWorldTools(server, w);
    {
        // Running, but nothing steps: answered from poll at the deadline with the steps measured so far (0)
        Poller poller(server);
        Json m = callTool(server, "measure_performance", obj(R"({"steps": 10})"));
        ASSERT_FALSE(m["result"]["isError"].bool_value()) << m.dump();
        EXPECT_TRUE(toolJson(m)["stopped_early"].bool_value());
    }
    {
        // Resets every 3 steps: one falls into the 300-step measurement whenever the request starts
        int steps = 0;
        Poller poller(server,
                      [&]()
                      {
                          w.step();
                          if (++steps % 3 == 0) w.profiler().reset();
                      });
        Json m = callTool(server, "measure_performance", obj(R"({"steps": 300})"));
        ASSERT_TRUE(m["result"]["isError"].bool_value());
        EXPECT_NE(m["result"]["content"][0]["text"].string_value().find("reset"), std::string::npos) << m.dump();
    }
}

// Regression (CodeRabbit): a step listener before the API's destroyed a moved entity and reused its slot; the API took
// the handle afterwards and reported the replacement as moved
TEST(WorldApi, MovesKeepTheHandleOfTheStep)
{
    World w(apiSetup());
    Entity mover = w.create();
    Entity replacement;
    w.addStepListener(
        [&]()
        {
            if (!w.alive(mover)) return false;
            w.destroy(mover);
            replacement = w.create();
            return false;
        });
    WorldApi api(w);
    Json cursor = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})")).value["cursor"];
    w.get<Transform>(mover).setPosition(Vec3(1, 0, 0));
    w.step();
    ASSERT_EQ(replacement.index, mover.index);
    ApiResult r = api.call("changes_since", Json(Json::object{{"cursor", cursor}, {"include_moved", true}}));
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_TRUE(r.value["moved"].array_items().empty()) << r.value.dump();
}

// Regression (CodeRabbit): an entity that moved again while a window was paged was deferred out of the window
TEST(WorldApi, AnEntityMovingAgainDuringPagingStaysInItsWindow)
{
    World w(apiSetup());
    WorldApi api(w);
    std::vector<Entity> e;
    for (int i = 0; i < 3; ++i) e.push_back(w.create());
    Json cursor = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})")).value["cursor"];
    w.get<Transform>(e[0]).setPosition(Vec3(1, 0, 0));
    w.get<Transform>(e[2]).setPosition(Vec3(1, 0, 0));
    w.step();
    ApiResult p1 =
        api.call("changes_since", Json(Json::object{{"cursor", cursor}, {"include_moved", true}, {"limit", 1}}));
    ASSERT_TRUE(p1.value["moved_more"].bool_value());
    // e[2] moves again before page two: still reported in this window
    w.get<Transform>(e[2]).setPosition(Vec3(2, 0, 0));
    w.step();
    ApiResult p2 = api.call("changes_since",
                            Json(Json::object{{"cursor", p1.value["cursor"]}, {"include_moved", true}, {"limit", 1}}));
    ASSERT_EQ(p2.value["moved"].array_items().size(), 1u);
    EXPECT_EQ(entityOf(p2.value["moved"][0]), e[2]);
}

namespace
{
enum class OddClass : uint16_t
{
    Odd = kFirstGameClass + 1,
};
// Throws something that is not a std::exception
class Odd final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(OddClass::Odd);
    static constexpr const char* name = "Odd";
    void onSpawn() override { throw 42; }
};
}  // namespace

// Regression (CodeRabbit): only std::exception was caught; any exception now becomes an indexed batch failure
TEST(WorldApi, AnyExceptionBecomesAnErrorResult)
{
    WorldSetup s = apiSetup();
    s.classes    = [](ClassList& c)
    {
        c.add<Dummy>();
        c.add<Odd>();
    };
    World w(s);
    WorldApi api(w);
    ApiResult r = api.call("apply", obj(R"({"operations": [{"op": "spawn", "position": [0, 0, 0]},
                                                         {"op": "spawn", "class": "Odd", "position": [0, 0, 0]}]})"));
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.value["applied"].number_value(), 1);
}

// Regression (CodeRabbit): with retained samples and no new step, the measurement returned the old history
TEST(WorldTools, AMeasurementWithoutStepsIsEmpty)
{
    World w(apiSetup());
    w.step(5);
    McpServer server("test");
    server.toolTimeoutSeconds = 1;
    auto api                  = registerWorldTools(server, w);
    Poller poller(server);
    Json m = callTool(server, "measure_performance", obj(R"({"steps": 10})"));
    ASSERT_FALSE(m["result"]["isError"].bool_value()) << m.dump();
    EXPECT_EQ(toolJson(m)["samples"].number_value(), 0);
    EXPECT_TRUE(toolJson(m)["stopped_early"].bool_value());
}

// Regression (CodeRabbit): with a page of 1, an entity that moved in a window but kept moving (both of its records
// outside the window by the time its page came) was never reported; window members are a snapshot now
TEST(WorldApi, EveryEntityOfAWindowIsDelivered)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity a = w.create(), b = w.create();
    Json cursor = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})")).value["cursor"];
    std::set<uint32_t> reported;
    for (int round = 0; round < 6; ++round)
    {
        // Both move every step, several times between pages
        for (int s = 0; s < 3; ++s)
        {
            w.get<Transform>(a).setPosition(Vec3(double(round * 3 + s + 1), 0, 0));
            w.get<Transform>(b).setPosition(Vec3(double(round * 3 + s + 1), 0, 0));
            w.step();
        }
        ApiResult r =
            api.call("changes_since", Json(Json::object{{"cursor", cursor}, {"include_moved", true}, {"limit", 1}}));
        ASSERT_TRUE(r.ok) << r.error;
        for (auto& m : r.value["moved"].array_items()) reported.insert(uint32_t(m["index"].number_value()));
        cursor = r.value["cursor"];
    }
    EXPECT_TRUE(reported.count(a.index));
    EXPECT_TRUE(reported.count(b.index)) << "b was pushed out of every window";
}

// Regression (CodeRabbit): two cursors paging the same window: the first to finish erased the snapshot the second still
// needed; snapshots are shared and kept now
TEST(WorldApi, TwoCursorsPageTheSameWindow)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity a = w.create(), b = w.create();
    Json start = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})")).value["cursor"];
    w.get<Transform>(a).setPosition(Vec3(1, 0, 0));
    w.get<Transform>(b).setPosition(Vec3(1, 0, 0));
    w.step();
    auto page = [&](const Json& c)
    { return api.call("changes_since", Json(Json::object{{"cursor", c}, {"include_moved", true}, {"limit", 1}})); };
    ApiResult x1 = page(start), y1 = page(start);
    ApiResult x2 = page(x1.value["cursor"]);  // x finishes its window
    // b moves twice before y continues
    for (int s = 0; s < 2; ++s)
    {
        w.get<Transform>(b).setPosition(Vec3(2.0 + s, 0, 0));
        w.step();
    }
    ApiResult y2 = page(y1.value["cursor"]);
    ASSERT_TRUE(y2.ok) << y2.error;
    ASSERT_EQ(y2.value["moved"].array_items().size(), 1u) << y2.value.dump();
    EXPECT_EQ(entityOf(y2.value["moved"][0]), b);
    EXPECT_FALSE(y2.value["moves_truncated"].bool_value());
}

// The request's time budget also ends a batch between operations
TEST(WorldApi, ApplyStopsAtTheTimeBudget)
{
    World w(apiSetup());
    WorldApi api(w);
    api.setTimeBudget(-1);  // already over
    ApiResult r = api.call("apply", obj(R"({"operations": [{"op": "world_status"}]})"));
    EXPECT_FALSE(r.ok);
    EXPECT_TRUE(r.value["stopped_early"].bool_value());
    EXPECT_EQ(r.value["applied"].number_value(), 0);
}

// Regression (CodeRabbit): a window snapshot rebuilt after eviction kept no truncation state, so a second cursor
// continuing it was not told that entities may be missing
TEST(WorldApi, EveryCursorOfAnEvictedWindowIsToldItIsTruncated)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity a = w.create(), b = w.create();
    auto moveBoth = [&]()
    {
        w.get<Transform>(a).setPosition(w.read<Transform>(a).position() + Vec3(1, 0, 0));
        w.get<Transform>(b).setPosition(w.read<Transform>(b).position() + Vec3(1, 0, 0));
        w.step();
    };
    auto page = [&](const Json& c)
    { return api.call("changes_since", Json(Json::object{{"cursor", c}, {"include_moved", true}, {"limit", 1}})); };
    Json start = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})")).value["cursor"];
    moveBoth();
    ApiResult x = page(start), y = page(start);
    ASSERT_TRUE(x.value["moved_more"].bool_value());
    // 16 newer unfinished windows evict this one
    for (int i = 0; i < 16; ++i)
    {
        Json c = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})")).value["cursor"];
        moveBoth();
        ASSERT_TRUE(page(c).value["moved_more"].bool_value());
    }
    EXPECT_TRUE(page(x.value["cursor"]).value["moves_truncated"].bool_value());
    EXPECT_TRUE(page(y.value["cursor"]).value["moves_truncated"].bool_value());
}

// Repeated measurements on a world that does not step: each is an empty window and leaves no listener behind
TEST(WorldTools, RepeatedMeasurementsWithoutStepsLeaveNoListeners)
{
    World w(apiSetup());
    w.step(3);
    McpServer server("test");
    server.toolTimeoutSeconds = 0.4;
    auto api                  = registerWorldTools(server, w);
    const size_t listeners    = w.stepListenerCount();
    {
        Poller poller(server);
        for (int i = 0; i < 3; ++i)
        {
            Json m = callTool(server, "measure_performance", obj(R"({"steps": 10})"));
            ASSERT_FALSE(m["result"]["isError"].bool_value()) << m.dump();
            EXPECT_EQ(toolJson(m)["samples"].number_value(), 0);
        }
    }
    // Read after the polling thread stopped
    EXPECT_EQ(w.stepListenerCount(), listeners);
}

// Regression (CodeRabbit): a call that waited in the queue for more than half the timeout got a fresh budget and
// answered after its caller had timed out; the handler now works to the request's real deadline
TEST(WorldTools, ALongQueueWaitStillAnswersInTime)
{
    WorldSetup setup = apiSetup();
    setup.systems    = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "slow",
              [](World&, double) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); });
    };
    World w(setup);
    w.setPaused(true);
    McpServer server("test");
    server.toolTimeoutSeconds = 1.0;
    auto api                  = registerWorldTools(server, w);
    Json response;
    // 5000 steps of 1 ms: far more than the 0.3 s left after the queue wait
    std::thread caller([&]() { response = callTool(server, "step", obj(R"({"n": 5000})")); });
    // The main loop is busy for 0.6 s before it polls
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
    {
        server.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    caller.join();
    ASSERT_FALSE(response["result"]["isError"].bool_value()) << response.dump();
    EXPECT_TRUE(toolJson(response)["stopped_early"].bool_value());
}

// ---------------------------------------------------------------------------------------------------------------------
// Snapshots and data (S11)

TEST(WorldApi, SnapshotsSaveDiffAndRollBack)
{
    World w(apiSetup());
    WorldApi api(w);
    const Entity d = entityOf(api.call("spawn", obj(R"({"class": "Dummy", "position": [1, 0, 2]})")).value["entity"]);
    ApiResult s    = api.call("save_snapshot", obj(R"({"name": "start"})"));
    ASSERT_TRUE(s.ok) << s.error;
    EXPECT_GT(s.value["bytes"].number_value(), 0);
    EXPECT_EQ(s.value["hash"].string_value().size(), 16u);
    // A name is not overwritten by accident
    EXPECT_FALSE(api.call("save_snapshot", obj(R"({"name": "start"})")).ok);
    EXPECT_TRUE(api.call("save_snapshot", obj(R"({"name": "start", "overwrite": true})")).ok);
    // An experiment: change, spawn, destroy
    ASSERT_TRUE(api.call("set_component", Json::object{{"entity", WorldApi::entityJson(d)},
                                                       {"component", "Health"},
                                                       {"values", Json::object{{"hp", 1}}}})
                    .ok);
    ASSERT_TRUE(api.call("spawn", obj(R"({"position": [9, 0, 9]})")).ok);
    ApiResult diff = api.call("diff_snapshot", obj(R"({"name": "start"})"));
    ASSERT_TRUE(diff.ok) << diff.error;
    EXPECT_EQ(diff.value["summary"]["components"]["Health"].number_value(), 1);
    EXPECT_EQ(diff.value["summary"]["kinds"]["only_in_b"].number_value(), 1);
    bool field = false;
    for (auto& x : diff.value["differences"].array_items())
        if (x["kind"] == "field")
        {
            field = true;
            EXPECT_EQ(x["component"].string_value(), "Health");
            EXPECT_EQ(x["field"].string_value(), "hp");
            EXPECT_EQ(x["b"].number_value(), 1);
        }
    EXPECT_TRUE(field);
    // Filters and paging
    EXPECT_EQ(api.call("diff_snapshot", obj(R"({"name": "start", "components": ["Health"]})")).value["total"], 1);
    ApiResult page = api.call("diff_snapshot", obj(R"({"name": "start", "limit": 1})"));
    ASSERT_TRUE(page.ok);
    EXPECT_EQ(page.value["differences"].array_items().size(), 1u);
    EXPECT_EQ(page.value["next"].number_value(), 1);
    // Against another snapshot
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "after"})")).ok);
    EXPECT_EQ(api.call("diff_snapshot", obj(R"({"name": "after", "against": "after"})")).value["total"], 0);
    // Roll back, paused
    ApiResult l = api.call("load_snapshot", obj(R"({"name": "start", "paused": true})"));
    ASSERT_TRUE(l.ok) << l.error;
    EXPECT_TRUE(l.value["paused"].bool_value());
    EXPECT_EQ(l.value["world_generation"].number_value(), 1);
    EXPECT_TRUE(l.value["hook_errors"].array_items().empty());
    EXPECT_EQ(w.read<Health>(d).hp, Health().hp);
    EXPECT_EQ(api.call("diff_snapshot", obj(R"({"name": "start"})")).value["total"], 0);
    EXPECT_EQ(api.call("world_status", Json()).value["world_generation"].number_value(), 1);
    // The game object came back
    EXPECT_TRUE(w.read<Behaviour>(d).object);
    // List, delete, unknown names
    EXPECT_EQ(api.call("list_snapshots", Json()).value["snapshots"].array_items().size(), 2u);
    EXPECT_TRUE(api.call("delete_snapshot", obj(R"({"name": "after"})")).ok);
    EXPECT_FALSE(api.call("delete_snapshot", obj(R"({"name": "after"})")).ok);
    EXPECT_FALSE(api.call("load_snapshot", obj(R"({"name": "nope"})")).ok);
    EXPECT_FALSE(api.call("diff_snapshot", obj(R"({"name": "start", "against": "nope"})")).ok);
}

TEST(WorldApi, SnapshotLimits)
{
    World w(apiSetup());
    WorldApi api(w);
    for (int i = 0; i < 16; ++i)
        ASSERT_TRUE(api.call("save_snapshot", Json::object{{"name", "s" + std::to_string(i)}}).ok);
    ApiResult r = api.call("save_snapshot", obj(R"({"name": "one_more"})"));
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("at most 16"), std::string::npos) << r.error;
    // Overwriting one is fine at the limit
    EXPECT_TRUE(api.call("save_snapshot", obj(R"({"name": "s3", "overwrite": true})")).ok);
    EXPECT_FALSE(api.call("save_snapshot", obj(R"({"name": ""})")).ok);
}

// A load (from anywhere) is one journal entry; cursors from before it stay valid and say reloaded
TEST(WorldApi, LoadsAppearAsOneJournalEntry)
{
    World w(apiSetup());
    WorldApi api(w);
    const auto blob = w.save();
    // More entities than the journal holds: a load must not flood it
    for (int i = 0; i < 1000; ++i) w.create();
    Json cursor         = api.call("changes_since", obj(R"({"cursor": "now", "include_moved": true})")).value["cursor"];
    const Entity before = w.create();  // a change before the load is still reported
    w.step(3);
    ASSERT_TRUE(w.tryLoad(blob).ok());
    ApiResult c = api.call("changes_since", Json::object{{"cursor", cursor}, {"include_moved", true}});
    ASSERT_TRUE(c.ok) << c.error;
    EXPECT_TRUE(c.value["reloaded"].bool_value());
    const auto& changes = c.value["changes"].array_items();
    ASSERT_EQ(changes.size(), 2u) << c.value.dump();
    EXPECT_EQ(changes[0]["kind"].string_value(), "created");
    EXPECT_EQ(entityOf(changes[0]["entity"]), before);
    EXPECT_EQ(changes[1]["kind"].string_value(), "world_loaded");
    EXPECT_EQ(changes[1]["from_step"].number_value(), 3);
    EXPECT_EQ(changes[1]["step"].number_value(), 0);
    EXPECT_EQ(changes[1]["world_generation"].number_value(), 1);
    // The next call with the new cursor continues normally
    Json next = c.value["cursor"];
    w.create();
    c = api.call("changes_since", Json::object{{"cursor", next}});
    ASSERT_TRUE(c.ok) << c.error;
    EXPECT_FALSE(c.value["reloaded"].bool_value());
    EXPECT_EQ(c.value["changes"].array_items().size(), 1u);
}

// rebind moves the API to another world: snapshots and cursors stay
TEST(WorldApi, RebindKeepsSnapshotsAndCursors)
{
    World a(apiSetup());
    WorldApi api(a);
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "x"})")).ok);
    Json cursor = api.call("changes_since", obj(R"({"cursor": "now"})")).value["cursor"];
    World b(apiSetup());
    b.step(4);
    api.rebind(b);
    EXPECT_EQ(api.world(), &b);
    ApiResult c = api.call("changes_since", Json::object{{"cursor", cursor}});
    ASSERT_TRUE(c.ok) << c.error;
    ASSERT_EQ(c.value["changes"].array_items().size(), 1u);
    EXPECT_EQ(c.value["changes"][0]["kind"].string_value(), "world_loaded");
    EXPECT_TRUE(api.call("load_snapshot", obj(R"({"name": "x"})")).ok);
    EXPECT_EQ(b.stepIndex(), 0u);
    // The old world is no longer observed: its changes do not appear
    a.create();
    a.step();
    c = api.call("changes_since", Json::object{{"cursor", c.value["cursor"]}});
    ASSERT_TRUE(c.ok) << c.error;
    for (auto& x : c.value["changes"].array_items()) EXPECT_EQ(x["kind"].string_value(), "world_loaded");
}

TEST(WorldApi, DataTools)
{
    World w(apiSetup());
    WorldApi api(w);
    ApiResult d = api.call("describe_world", Json());
    ASSERT_EQ(d.value["data_types"].array_items().size(), 1u);
    EXPECT_EQ(d.value["data_types"][0]["file"].string_value(), "creature_stats.json");
    EXPECT_EQ(d.value["save_format_version"].number_value(), 1);
    ApiResult names = api.call("get_data", obj(R"({"type": "CreatureStats"})"));
    ASSERT_TRUE(names.ok) << names.error;
    EXPECT_EQ(names.value["entries"].array_items().size(), 2u);
    ApiResult wolf = api.call("get_data", obj(R"({"type": "CreatureStats", "name": "wolf"})"));
    ASSERT_TRUE(wolf.ok);
    EXPECT_EQ(wolf.value["values"]["speed"].number_value(), 6.5);
    EXPECT_FALSE(api.call("get_data", obj(R"({"type": "CreatureStats", "name": "cat"})")).ok);
    EXPECT_FALSE(api.call("get_data", obj(R"({"type": "Nope"})")).ok);
    ApiResult r = api.call("reload_data", obj(R"({"reconfigure": false})"));
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_EQ(r.value["entries"].number_value(), 2);
    EXPECT_EQ(w.loadCount(), 0u);
    r = api.call("reload_data", Json());
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_EQ(w.loadCount(), 1u);
}

TEST(WorldApi, SnapshotFiles)
{
    World w(apiSetup());
    WorldApi api(w);
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "a"})")).ok);
    // No directory: refused
    EXPECT_FALSE(api.call("export_snapshot", obj(R"({"name": "a", "file": "a.sgws"})")).ok);
    api.setSnapshotDirectory(::testing::TempDir());
    // Only plain file names
    EXPECT_FALSE(api.call("export_snapshot", obj(R"({"name": "a", "file": "../a.sgws"})")).ok);
    EXPECT_FALSE(api.call("export_snapshot", obj(R"({"name": "a", "file": "dir/a.sgws"})")).ok);
    ASSERT_TRUE(api.call("export_snapshot", obj(R"({"name": "a", "file": "saiga_api_a.sgws", "overwrite": true})")).ok);
    // An existing file is kept unless overwrite is set; nothing partial is left behind
    EXPECT_FALSE(api.call("export_snapshot", obj(R"({"name": "a", "file": "saiga_api_a.sgws"})")).ok);
    for (const auto& f : std::filesystem::directory_iterator(::testing::TempDir()))
        EXPECT_EQ(f.path().filename().string().find(".saiga-export-"), std::string::npos) << f.path();
    // Windows device names are refused
    EXPECT_FALSE(api.call("export_snapshot", obj(R"({"name": "a", "file": "NUL.bin"})")).ok);
    EXPECT_FALSE(api.call("export_snapshot", obj(R"({"name": "a", "file": "com1"})")).ok);
    ApiResult i = api.call("import_snapshot", obj(R"({"name": "b", "file": "saiga_api_a.sgws"})"));
    ASSERT_TRUE(i.ok) << i.error;
    // Its hash is unknown (not a fake 0)
    const ApiResult listed = api.call("list_snapshots", Json());
    for (auto& x : listed.value["snapshots"].array_items())
    {
        if (x["name"] == "b")
        {
            EXPECT_TRUE(x["hash"].is_null());
        }
    }
    EXPECT_EQ(api.call("diff_snapshot", obj(R"({"name": "a", "against": "b"})")).value["total"], 0);
    EXPECT_FALSE(api.call("import_snapshot", obj(R"({"name": "c", "file": "missing.sgws"})")).ok);
    // Read-only observers cannot load, import or export
    WorldApi observer(w, false);
    EXPECT_TRUE(observer.call("save_snapshot", obj(R"({"name": "o"})")).ok);
    EXPECT_TRUE(observer.call("diff_snapshot", obj(R"({"name": "o"})")).ok);
    EXPECT_FALSE(observer.call("load_snapshot", obj(R"({"name": "o"})")).ok);
    EXPECT_FALSE(observer.call("reload_data", Json()).ok);
    std::remove((::testing::TempDir() + "/saiga_api_a.sgws").c_str());
}

// A running measurement fails when the world is loaded meanwhile (its samples would mix two worlds)
TEST(WorldTools, MeasurementFailsOnALoad)
{
    World w(apiSetup());
    McpServer server("test");
    auto api        = registerWorldTools(server, w);
    const auto blob = w.save();
    int steps       = 0;
    Poller poller(server,
                  [&]()
                  {
                      w.step();
                      if (++steps == 5) w.tryLoad(blob);
                  });
    Json m = callTool(server, "measure_performance", obj(R"({"steps": 300})"));
    ASSERT_TRUE(m["result"]["isError"].bool_value()) << m.dump();
    EXPECT_NE(m["result"]["content"][0]["text"].string_value().find("loaded"), std::string::npos) << m.dump();
}

// ---------------------------------------------------------------------------------------------------------------------
// Review regressions (S11 round 1)

// A throwing load listener registered before the API made a committed load look failed and hid it from the journal
TEST(WorldApi, AThrowingLoadListenerDoesNotHideTheLoad)
{
    World w(apiSetup());
    w.addLoadListener([](const LoadInfo&) { throw std::runtime_error("listener boom"); });
    WorldApi api(w);
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "s"})")).ok);
    Json cursor = api.call("changes_since", obj(R"({"cursor": "now"})")).value["cursor"];
    ApiResult l = api.call("load_snapshot", obj(R"({"name": "s"})"));
    ASSERT_TRUE(l.ok) << l.error;
    ASSERT_EQ(l.value["listener_errors"].array_items().size(), 1u);
    EXPECT_EQ(l.value["listener_errors"][0].string_value(), "listener boom");
    ApiResult c = api.call("changes_since", Json::object{{"cursor", cursor}});
    ASSERT_EQ(c.value["changes"].array_items().size(), 1u) << c.value.dump();
    EXPECT_EQ(c.value["changes"][0]["kind"].string_value(), "world_loaded");
}

// Changes made by another load listener (after the load's own records) were folded into the load and lost
TEST(WorldApi, ChangesOfOtherLoadListenersAreReported)
{
    World w(apiSetup());
    WorldApi api(w);
    Entity created;
    w.addLoadListener([&](const LoadInfo&) { created = w.create(); });
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "s"})")).ok);
    Json cursor = api.call("changes_since", obj(R"({"cursor": "now"})")).value["cursor"];
    ASSERT_TRUE(api.call("load_snapshot", obj(R"({"name": "s"})")).ok);
    ApiResult c         = api.call("changes_since", Json::object{{"cursor", cursor}});
    const auto& changes = c.value["changes"].array_items();
    ASSERT_EQ(changes.size(), 2u) << c.value.dump();
    EXPECT_EQ(changes[0]["kind"].string_value(), "world_loaded");
    EXPECT_EQ(changes[1]["kind"].string_value(), "created");
    EXPECT_EQ(entityOf(changes[1]["entity"]), created);
}

// rebind kept the last destroyed entity of the old world: a removal on the same handle in the new world was dropped;
// world_generation only grows
TEST(WorldApi, RebindStartsAFreshJournalState)
{
    World a(apiSetup()), b(apiSetup());
    const Entity x = a.create(), y = b.create();
    ASSERT_EQ(x, y);
    b.add<Health>(y);
    WorldApi api(a);
    a.destroy(x);
    const double generation = api.call("world_status", Json()).value["world_generation"].number_value();
    Json cursor             = api.call("changes_since", obj(R"({"cursor": "now"})")).value["cursor"];
    api.rebind(b);
    EXPECT_GT(api.call("world_status", Json()).value["world_generation"].number_value(), generation);
    b.remove<Health>(y);
    ApiResult c  = api.call("changes_since", Json::object{{"cursor", cursor}});
    bool removed = false;
    for (auto& ch : c.value["changes"].array_items()) removed = removed || ch["kind"] == "removed";
    EXPECT_TRUE(removed) << c.value.dump();
}

// The entity filter of diff_snapshot compares the generation too (a reused slot is another entity)
TEST(WorldApi, DiffEntityFilterUsesTheGeneration)
{
    World w(apiSetup());
    WorldApi api(w);
    const Entity e = w.create();
    w.add<Health>(e);
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "s"})")).ok);
    w.get<Health>(e).hp = 1;
    const Entity stale{e.index, e.generation + 1};
    EXPECT_EQ(
        api.call("diff_snapshot", Json::object{{"name", "s"}, {"entity", WorldApi::entityJson(e)}}).value["total"], 1);
    EXPECT_EQ(
        api.call("diff_snapshot", Json::object{{"name", "s"}, {"entity", WorldApi::entityJson(stale)}}).value["total"],
        0);
}

// Review regressions (S11 round 2)

TEST(WorldApi, ImportedSnapshotsLearnTheirHashAndFiltersAreChecked)
{
    World w(apiSetup());
    WorldApi api(w);
    api.setSnapshotDirectory(::testing::TempDir());
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "a"})")).ok);
    ASSERT_TRUE(
        api.call("export_snapshot", obj(R"({"name": "a", "file": "saiga_api_hash.sgws", "overwrite": true})")).ok);
    ASSERT_TRUE(api.call("import_snapshot", obj(R"({"name": "b", "file": "saiga_api_hash.sgws"})")).ok);
    ASSERT_TRUE(api.call("load_snapshot", obj(R"({"name": "b"})")).ok);
    const ApiResult listed = api.call("list_snapshots", Json());
    for (auto& x : listed.value["snapshots"].array_items())
    {
        if (x["name"] == "b")
        {
            EXPECT_EQ(x["hash"].string_value().size(), 16u);
        }
    }
    // A typo in the components filter is an error, not an empty result
    EXPECT_FALSE(api.call("diff_snapshot", obj(R"({"name": "a", "components": ["Healht"]})")).ok);
    // reload_data reports listener errors like load_snapshot
    EXPECT_TRUE(api.call("reload_data", Json()).value["listener_errors"].is_array());
    std::remove((::testing::TempDir() + "/saiga_api_hash.sgws").c_str());
}

// Records made before a load carry the step before it (they got the loaded step)
TEST(WorldApi, RecordsBeforeALoadKeepTheirStep)
{
    World w(apiSetup());
    WorldApi api(w);
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "s"})")).ok);
    w.step(4);
    Json cursor = api.call("changes_since", obj(R"({"cursor": "now"})")).value["cursor"];
    w.create();
    ASSERT_TRUE(api.call("load_snapshot", obj(R"({"name": "s"})")).ok);
    ApiResult c = api.call("changes_since", Json::object{{"cursor", cursor}});
    ASSERT_GE(c.value["changes"].array_items().size(), 2u);
    EXPECT_EQ(c.value["changes"][0]["kind"].string_value(), "created");
    EXPECT_EQ(c.value["changes"][0]["step"].number_value(), 4);
}

// Regression (CodeRabbit, S11): a load listener that rebinds the API made load_snapshot apply paused to the new world
// and report its numbers as the loaded ones
TEST(WorldApi, LoadSnapshotNoticesARebindInAListener)
{
    World a(apiSetup());
    World b(apiSetup());
    WorldApi api(a);
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "x"})")).ok);
    const int id      = a.addLoadListener([&](const LoadInfo&) { api.rebind(b); });
    const ApiResult r = api.call("load_snapshot", obj(R"({"name": "x", "paused": true})"));
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("rebound"), std::string::npos) << r.error;
    EXPECT_FALSE(b.isPaused());
    EXPECT_EQ(api.world(), &b);
    a.removeLoadListener(id);
}

// Regression (CodeRabbit, S11): a listener that replaced an imported snapshot under the same name gave the replacement
// the hash of the bytes loaded before
TEST(WorldApi, ImportedSnapshotLearnsOnlyItsOwnHash)
{
    World w(apiSetup());
    WorldApi api(w);
    api.setSnapshotDirectory(::testing::TempDir());
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "one"})")).ok);
    ASSERT_TRUE(
        api.call("export_snapshot", obj(R"({"name": "one", "file": "saiga_rev_1.sgws", "overwrite": true})")).ok);
    w.create();
    w.step();
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "two"})")).ok);
    ASSERT_TRUE(
        api.call("export_snapshot", obj(R"({"name": "two", "file": "saiga_rev_2.sgws", "overwrite": true})")).ok);
    ASSERT_TRUE(api.call("import_snapshot", obj(R"({"name": "b", "file": "saiga_rev_1.sgws"})")).ok);
    bool replaced = false;
    const int id  = w.addLoadListener(
        [&](const LoadInfo&)
        {
            if (replaced) return;
            replaced = true;
            EXPECT_TRUE(
                api.call("import_snapshot", obj(R"({"name": "b", "file": "saiga_rev_2.sgws", "overwrite": true})")).ok);
        });
    ASSERT_TRUE(api.call("load_snapshot", obj(R"({"name": "b"})")).ok);
    w.removeLoadListener(id);
    const ApiResult first = api.call("list_snapshots", Json());
    for (auto& x : first.value["snapshots"].array_items())
        if (x["name"] == "b") EXPECT_TRUE(x["hash"].is_null()) << x.dump();
    // Loaded without a replacement, it learns its hash
    ASSERT_TRUE(api.call("load_snapshot", obj(R"({"name": "b"})")).ok);
    const ApiResult second = api.call("list_snapshots", Json());
    for (auto& x : second.value["snapshots"].array_items())
        if (x["name"] == "b") EXPECT_FALSE(x["hash"].is_null()) << x.dump();
}

// Regression (CodeRabbit, S11): save_snapshot (and diff_snapshot against the live world) from inside a game hook
// reached World::save's assert instead of returning a failure
namespace
{
WorldApi* g_hookApi = nullptr;
std::vector<ApiResult> g_hookResults;
enum class HookSaveClass : uint16_t
{
    Saver = kFirstGameClass + 1,
};
class Saver final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(HookSaveClass::Saver);
    static constexpr const char* name = "Saver";
    void onSpawn() override
    {
        g_hookResults.push_back(g_hookApi->call("save_snapshot", obj(R"({"name": "in_hook"})")));
        g_hookResults.push_back(g_hookApi->call("diff_snapshot", obj(R"({"name": "before"})")));
    }
};
}  // namespace

TEST(WorldApi, SnapshotSavesAreRefusedInsideGameHooks)
{
    WorldSetup s = apiSetup();
    s.classes    = [](ClassList& c)
    {
        c.add<Dummy>();
        c.add<Saver>();
    };
    World w(s);
    WorldApi api(w);
    g_hookApi = &api;
    g_hookResults.clear();
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "before"})")).ok);
    w.spawn<Saver>(Vec3::Zero());
    g_hookApi = nullptr;
    ASSERT_EQ(g_hookResults.size(), 2u);
    for (const ApiResult& r : g_hookResults)
    {
        EXPECT_FALSE(r.ok);
        EXPECT_NE(r.error.find("not inside a game hook"), std::string::npos) << r.error;
    }
    EXPECT_TRUE(api.call("save_snapshot", obj(R"({"name": "after"})")).ok);
}

// Regression (CodeRabbit, S11): an imported snapshot learned the hash of the world after the load listeners changed it
TEST(WorldApi, ImportedSnapshotLearnsTheHashBeforeTheListeners)
{
    World w(apiSetup());
    WorldApi api(w);
    api.setSnapshotDirectory(::testing::TempDir());
    ASSERT_TRUE(api.call("save_snapshot", obj(R"({"name": "one"})")).ok);
    const ApiResult saved = api.call("list_snapshots", Json());
    std::string hashOne;
    for (auto& x : saved.value["snapshots"].array_items())
        if (x["name"] == "one") hashOne = x["hash"].string_value();
    ASSERT_TRUE(
        api.call("export_snapshot", obj(R"({"name": "one", "file": "saiga_hash_1.sgws", "overwrite": true})")).ok);
    ASSERT_TRUE(api.call("import_snapshot", obj(R"({"name": "b", "file": "saiga_hash_1.sgws"})")).ok);
    const int id = w.addLoadListener([&](const LoadInfo&) { w.create(); });
    ASSERT_TRUE(api.call("load_snapshot", obj(R"({"name": "b"})")).ok);
    w.removeLoadListener(id);
    const ApiResult listed = api.call("list_snapshots", Json());
    for (auto& x : listed.value["snapshots"].array_items())
        if (x["name"] == "b") EXPECT_EQ(x["hash"].string_value(), hashOne) << x.dump();
}
