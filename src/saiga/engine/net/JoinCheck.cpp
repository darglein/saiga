/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/JoinCheck.h"

#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
JoinRequest joinRequestOf(const World& w)
{
    JoinRequest r;
    r.protocolVersion = kProtocolVersion;
    r.schemaHash      = w.netSchemaHash();
    r.sectionHashes   = w.netSchema().sectionHashes();
    r.renderTableHash = w.renderTableHash();
    r.seed            = w.generationSeed();
    r.environment     = w.environmentSetup().enabled;
    r.nonLegacy       = w.generation().nonLegacy();
    for (const GeneratorList::Entry* e : w.generators().byName())
    {
        JoinRequest::Generator g;
        g.name    = e->generator->name();
        g.version = e->generator->version();
        for (const GeneratorSlot& s : e->slots) g.slots.push_back(s.name);
        r.generators.push_back(std::move(g));
    }
    return r;
}

std::optional<std::pair<JoinRejectReason, std::string>> checkJoinRequest(const World& host, const JoinRequest& r)
{
    using Result = std::pair<JoinRejectReason, std::string>;
    if (r.protocolVersion != kProtocolVersion)
        return Result{JoinRejectReason::ProtocolVersion, "protocol version " + std::to_string(r.protocolVersion) +
                                                             ", the host's " + std::to_string(kProtocolVersion)};
    // Before the schema (D5): the environment flag changes the schema too
    const bool env = host.environmentSetup().enabled;
    if (r.environment != env)
        return Result{JoinRejectReason::Environment,
                      std::string("environment: the host's world has WorldTime and Weather ") +
                          (env ? "enabled, this client's not" : "disabled, this client's enabled")};
    const auto sections = host.netSchema().sectionHashes();
    if (r.schemaHash != host.netSchemaHash() || r.sectionHashes != sections)
    {
        std::string differing;
        for (size_t i = 0; i < kNetSchemaSections; ++i)
            if (r.sectionHashes[i] != sections[i])
                differing += (differing.empty() ? "" : ", ") + std::string(netSchemaSectionName(NetSchemaSection(i)));
        return Result{JoinRejectReason::Schema,
                      "schema: " + (differing.empty() ? std::string("the hash") : differing) + " differ"};
    }
    if (r.renderTableHash != host.renderTableHash())
        return Result{JoinRejectReason::RenderTables, "render name tables differ (meshes, materials, aliases)"};
    if (r.seed != host.generationSeed())
        return Result{JoinRejectReason::Seed,
                      "seed " + std::to_string(r.seed) + ", the host's " + std::to_string(host.generationSeed())};
    // Generators: the same names, the same non-legacy flag, every host version runnable, the host's slots a prefix
    const GenerationPlans& plans = host.generation().plans();
    const auto hostGens          = host.generators().byName();
    auto names                   = [](const std::vector<std::string>& v)
    {
        std::string s;
        for (const std::string& n : v) s += (s.empty() ? "" : ", ") + n;
        return s;
    };
    std::vector<std::string> hostNames, clientNames;
    for (const GeneratorList::Entry* e : hostGens) hostNames.push_back(e->generator->name());
    for (const JoinRequest::Generator& g : r.generators) clientNames.push_back(g.name);
    if (hostNames != clientNames)
        return Result{JoinRejectReason::Generators,
                      "generators: the host's [" + names(hostNames) + "], this client's [" + names(clientNames) + "]"};
    if (r.nonLegacy != host.generation().nonLegacy())
        return Result{JoinRejectReason::Generators, "generators: one side generates legacy terrain only"};
    for (size_t i = 0; i < hostGens.size(); ++i)
    {
        const std::string& name = hostNames[i];
        const uint16_t current  = plans.current.at(name);
        if (r.generators[i].version < current)
            return Result{JoinRejectReason::Generators,
                          "generators: " + name + " v" + std::to_string(r.generators[i].version) +
                              " on this client, the host places v" + std::to_string(current)};
        const std::vector<GeneratorSlot>& slots = hostGens[i]->slots;
        bool prefix                             = slots.size() <= r.generators[i].slots.size();
        for (size_t k = 0; prefix && k < slots.size(); ++k) prefix = slots[k].name == r.generators[i].slots[k];
        if (!prefix)
            return Result{JoinRejectReason::Generators,
                          "generators: the host's slot table of " + name + " is not a prefix of this client's"};
    }
    return std::nullopt;
}

bool adoptablePins(const World& client, const GenerationInfo& host, GenerationPins& pins, std::string& error)
{
    pins = GenerationPins();
    if (host.seed != client.generationSeed())
        return (error = "the host's seed " + std::to_string(host.seed) + ", this client's " +
                        std::to_string(client.generationSeed()),
                false);
    const auto gens = client.generators().byName();
    if (gens.size() != host.generators.size())
        return (error = "the host has " + std::to_string(host.generators.size()) + " generators, this client " +
                        std::to_string(gens.size()),
                false);
    for (size_t i = 0; i < gens.size(); ++i)
    {
        const GenerationInfo::Generator& h = host.generators[i];
        const std::string name             = gens[i]->generator->name();
        if (h.name != name) return (error = "the host's generator " + h.name + ", this client's " + name, false);
        const uint16_t version = gens[i]->generator->version();
        if (h.pinned == 0 || h.pinned > h.current || h.current > version)
            return (error = "generator " + name + ": the host's v" + std::to_string(h.pinned) + " pinned, v" +
                            std::to_string(h.current) + " current, this client has v" + std::to_string(version),
                    false);
        const std::vector<GeneratorSlot>& slots = gens[i]->slots;
        bool prefix                             = h.slots.size() <= slots.size();
        for (size_t k = 0; prefix && k < h.slots.size(); ++k)
            prefix = h.slots[k].name == slots[k].name && h.slots[k].flags == slots[k].flags;
        if (!prefix)
            return (error = "generator " + name + ": the host's slot table is not a prefix of this client's", false);
        pins.pinned[name]  = h.pinned;
        pins.current[name] = h.current;
    }
    return true;
}

}  // namespace Engine
}  // namespace Saiga
