/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The JSON visitors, the world API and its MCP tools (docs/engine_design.md §16.2; stage S10)

#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/ecs/VisitJson.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/tools/WorldTools.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "mcp_test_helper.h"

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
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

std::vector<uint32_t> g_actions;
void sysActions(World& w, double)
{
    for (const ActionEvent& a : w.events.read<ActionEvent>()) g_actions.push_back(a.action);
}

WorldSetup apiSetup()
{
    WorldSetup s;
    s.components = [](Registry& r) { r.registerGlobal<Weather>(Replicate::Host); };
    s.classes    = [](ClassList& c) { c.add<Dummy>(); };
    s.systems    = [](SystemList& l) { l.add(Phase::Gameplay, "actions", sysActions); };
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
    ApiResult c0 = api.call("changes_since", obj(R"({"cursor": 0})"));
    ASSERT_TRUE(c0.ok);
    const double cursor = c0.value["cursor"].number_value();
    Entity a            = w.create();
    w.add<Health>(a);
    w.step();
    w.get<Transform>(a).setPosition(Vec3(1, 0, 0));
    w.step();
    w.destroy(a);
    ApiResult c1 = api.call("changes_since", Json(Json::object{{"cursor", cursor}}));
    ASSERT_TRUE(c1.ok);
    std::vector<std::string> kinds;
    for (auto& c : c1.value["changes"].array_items()) kinds.push_back(c["kind"].string_value());
    EXPECT_EQ(kinds, (std::vector<std::string>{"created", "added", "moved", "destroyed", "removed"}));
    // Nothing new after the returned cursor
    EXPECT_TRUE(api.call("changes_since", Json(Json::object{{"cursor", c1.value["cursor"]}}))
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
    Entity b             = w.create();
    const std::string eb = entityString(b);
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
    EXPECT_EQ(w.read<Health>(b).hp, 3);
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
    EXPECT_EQ(toolJson(set)["hp"].number_value(), 12);
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
