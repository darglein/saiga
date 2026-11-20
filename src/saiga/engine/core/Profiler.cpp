/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Profiler: timings and counters per sample with ring histories.

#include "saiga/engine/core/Profiler.h"

#include "saiga/core/util/assert.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
Profiler::Profiler(int historySize) : capacity(std::max(1, historySize)) {}

int Profiler::scope(const std::string& path)
{
    for (size_t i = 0; i < scopes.size(); ++i)
        if (scopes[i].name == path) return int(i);
    const size_t slash = path.rfind('/');
    const int parent   = slash == std::string::npos ? -1 : scope(path.substr(0, slash));
    Series s;
    s.name   = path;
    s.parent = parent;
    s.ring.assign(size_t(capacity), 0.0);
    scopes.push_back(std::move(s));
    return int(scopes.size() - 1);
}

int Profiler::counter(const std::string& name)
{
    for (size_t i = 0; i < counters.size(); ++i)
        if (counters[i].name == name) return int(i);
    Series s;
    s.name = name;
    s.ring.assign(size_t(capacity), 0.0);
    counters.push_back(std::move(s));
    return int(counters.size() - 1);
}

void Profiler::add(int scope, double ms)
{
    scopes[size_t(scope)].current += ms;
}

void Profiler::setCounter(int counter, double value)
{
    counters[size_t(counter)].current = value;
}

void Profiler::addCounter(int counter, double value)
{
    counters[size_t(counter)].current += value;
}

void Profiler::push(Series& s)
{
    s.ring[size_t(head)] = s.current;
    s.current            = 0;
}

void Profiler::endSample()
{
    for (auto& s : scopes) push(s);
    for (auto& s : counters) push(s);
    head   = (head + 1) % capacity;
    filled = std::min(filled + 1, capacity);
    samples++;
}

void Profiler::discardSample()
{
    for (auto& s : scopes) s.current = 0;
    for (auto& s : counters) s.current = 0;
}

void Profiler::reset()
{
    for (auto& s : scopes) std::fill(s.ring.begin(), s.ring.end(), 0.0), s.current = 0;
    for (auto& s : counters) std::fill(s.ring.begin(), s.ring.end(), 0.0), s.current = 0;
    filled = 0;
    head   = 0;
    resets++;
}

std::vector<double> Profiler::scopeHistory(int scope) const
{
    std::vector<double> out;
    if (scope < 0 || scope >= int(scopes.size())) return out;
    const Series& s = scopes[size_t(scope)];
    for (int i = filled; i >= 1; --i) out.push_back(s.ring[size_t((head - i + capacity) % capacity)]);
    return out;
}

Profiler::Stats Profiler::stats(const Series& s, int lastSamples) const
{
    Stats st;
    const int n = lastSamples <= 0 ? filled : std::min(lastSamples, filled);
    if (n == 0) return st;
    std::vector<double> v;
    v.reserve(size_t(n));
    for (int i = 1; i <= n; ++i) v.push_back(s.ring[size_t((head - i + capacity) % capacity)]);
    double sum = 0;
    for (double x : v) sum += x;
    std::sort(v.begin(), v.end());
    // Nearest-rank percentiles
    auto rank  = [&](double q) { return v[size_t(std::min(double(n - 1), std::ceil(q * n) - 1))]; };
    st.samples = n;
    st.avg     = sum / n;
    st.min     = v.front();
    st.max     = v.back();
    st.p50     = rank(0.5);
    st.p95     = rank(0.95);
    return st;
}

Profiler::Stats Profiler::scopeStats(int scope, int lastSamples) const
{
    return stats(scopes[size_t(scope)], lastSamples);
}

Profiler::Stats Profiler::counterStats(int counter, int lastSamples) const
{
    return stats(counters[size_t(counter)], lastSamples);
}

json11::Json Profiler::report(int lastSamples) const
{
    using Json = json11::Json;
    // Times rounded to 1 ns (1e-6 ms): more digits are noise
    auto round6    = [](double v) { return std::round(v * 1e6) / 1e6; };
    auto statsJson = [&](const Stats& s)
    {
        return Json::object{{"samples", s.samples}, {"avg", round6(s.avg)}, {"min", round6(s.min)},
                            {"p50", round6(s.p50)}, {"p95", round6(s.p95)}, {"max", round6(s.max)}};
    };
    Json::array scopeList;
    for (size_t i = 0; i < scopes.size(); ++i)
    {
        const Stats s          = stats(scopes[i], lastSamples);
        Json::object o         = statsJson(s);
        o["scope"]             = scopes[i].name;
        const int parent       = scopes[i].parent;
        const double parentAvg = parent >= 0 ? stats(scopes[size_t(parent)], lastSamples).avg : 0;
        o["share_of_parent"]   = std::round((parent >= 0 && parentAvg > 0 ? s.avg / parentAvg : 1.0) * 1e4) / 1e4;
        scopeList.push_back(o);
    }
    Json::array counterList;
    for (const Series& c : counters)
    {
        Json::object o = statsJson(stats(c, lastSamples));
        o["counter"]   = c.name;
        counterList.push_back(o);
    }
    // The slowest leaf scopes (systems, command flushes) by average: where the time goes
    std::vector<std::pair<double, std::string>> leaves;
    for (size_t i = 0; i < scopes.size(); ++i)
    {
        bool leaf = true;
        for (const Series& other : scopes) leaf = leaf && other.parent != int(i);
        if (leaf) leaves.push_back({stats(scopes[i], lastSamples).avg, scopes[i].name});
    }
    std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    Json::array top;
    for (size_t i = 0; i < leaves.size() && i < 10; ++i)
        top.push_back(Json::object{{"scope", leaves[i].second}, {"avg", round6(leaves[i].first)}});
    const int n = lastSamples <= 0 ? filled : std::min(lastSamples, filled);
    return Json::object{{"unit", "ms"}, {"samples", n}, {"top", top}, {"scopes", scopeList}, {"counters", counterList}};
}

}  // namespace Engine
}  // namespace Saiga
