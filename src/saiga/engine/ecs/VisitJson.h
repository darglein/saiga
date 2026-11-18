/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/ecs/Visit.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>
#include <vector>

// JSON form of visit (docs/engine_design.md §7.2, for the world API and MCP; S11: dumpJson and data files).
//
//   bool, numbers          JSON bool / number; 64-bit integers beyond 2^53 as decimal strings (exact); non-finite
//                          doubles as "nan", "inf", "-inf"
//   enums                  the value name (EnumNames)
//   Entity                 {"index": i, "generation": g} (the reader also accepts "i:g")
//   PersistentId           the value (like uint64)
//   Vec2/3/4, Mat3/4       arrays (matrices column-major, like the binary form)
//   Quat                   [w, x, y, z]
//   std::vector, std::array  arrays (an array needs exactly N elements)
//   nested structs         objects
//
// Reading merges: keys that are missing keep the current values; unknown keys, wrong types, out-of-range integers and
// wrong array lengths are errors. finiteOnly (writes from the world API): nan and inf are errors, quaternions need a
// length > 0 and are normalized (unit ones stay bit-exact). fromJson first checks everything without writing (a dry run
// on a scratch copy of the structure), so a rejected document changes nothing.

namespace Saiga
{
namespace Engine
{
class JsonWriter
{
   public:
    using Json = json11::Json;

    template <class T>
    void operator()(const char* name, T& field)
    {
        object[name] = write(field);
    }
    template <class E, size_t N>
    void operator()(const char* name, E& field, const EnumNames<N>& names)
    {
        const size_t v = size_t(field);
        object[name]   = v < N ? Json(names.names[v]) : Json(int(v));
    }
    template <class T>
    void operator()(const char* name, T& field, const Quantize&)
    {
        (*this)(name, field);
    }

    template <class T>
    static Json write(const T& v)
    {
        using namespace VisitDetail;
        if constexpr (std::is_same_v<T, bool>)
            return Json(v);
        else if constexpr (std::is_enum_v<T>)
        {
            // Only reachable for a component that registration rejects (enums need EnumNames): written as a number so
            // the field check reports the real problem (MSVC stops at the first static_assert)
            return Json(double(int64_t(v)));
        }
        else if constexpr (isInteger<T>)
        {
            // Exact in a double below 2^53 (compared as integers: 2^53 + 1 would round to 2^53 in a double); larger
            // 64-bit values as strings
            constexpr int64_t kExact = int64_t(1) << 53;
            if constexpr (std::is_signed_v<T>)
            {
                if (int64_t(v) >= kExact || int64_t(v) <= -kExact) return Json(std::to_string(v));
            }
            else
            {
                if (uint64_t(v) >= uint64_t(kExact)) return Json(std::to_string(v));
            }
            return Json(double(v));
        }
        else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            if (std::isnan(v)) return Json("nan");
            if (std::isinf(v)) return Json(v > 0 ? "inf" : "-inf");
            return Json(double(v));
        }
        else if constexpr (std::is_same_v<T, std::string>)
            return Json(v);
        else if constexpr (std::is_same_v<T, Entity>)
            return Json(Json::object{{"index", double(v.index)}, {"generation", double(v.generation)}});
        else if constexpr (std::is_same_v<T, PersistentId>)
            return write(v.value);
        else if constexpr (isDoubleMath<T> || isFloatMath<T>)
        {
            Json::array a;
            for (int c = 0; c < int(T::ColsAtCompileTime); ++c)
                for (int r = 0; r < int(T::RowsAtCompileTime); ++r) a.push_back(write(v(r, c)));
            return a;
        }
        else if constexpr (isQuat<T>)
            return Json::array{write(v.w()), write(v.x()), write(v.y()), write(v.z())};
        else if constexpr (IsStdVector<T>::value || IsStdArray<T>::value)
        {
            Json::array a;
            for (const auto& e : v) a.push_back(write(e));
            return a;
        }
        else if constexpr (IsVisitStruct<T>::value)
        {
            JsonWriter w;
            const_cast<T&>(v).visit(w);  // visit is non-const by design
            return w.object;
        }
        else
        {
            static_assert(AlwaysFalse<T>::value, "JsonWriter: unsupported field type (see Visit.h)");
            return Json();
        }
    }

    Json::object object;
};

// Reads the fields present in a JSON object into a struct (see the header). Apply mode writes; dry-run mode only
// checks.
class JsonReader
{
   public:
    using Json = json11::Json;

    JsonReader(const Json& object, bool apply, bool finiteOnly = false) : apply(apply), finiteOnly(finiteOnly)
    {
        push(object, "");
    }

    bool ok() const { return error.empty(); }
    const std::string& errorMessage() const { return error; }

    template <class T>
    void operator()(const char* name, T& field)
    {
        frames.back().names.push_back(name);
        const Json* j = take(name);
        if (j) read(*j, field, path + name);
    }
    template <class E, size_t N>
    void operator()(const char* name, E& field, const EnumNames<N>& names)
    {
        frames.back().names.push_back(name);
        const Json* j = take(name);
        if (!j || !ok()) return;
        if (!j->is_string()) return fail(path + name, "expected one of the enum names");
        for (size_t i = 0; i < N; ++i)
            if (j->string_value() == names.names[i])
            {
                if (apply) field = E(i);
                return;
            }
        fail(path + name, "unknown enum value \"" + j->string_value() + "\"");
    }
    template <class T>
    void operator()(const char* name, T& field, const Quantize&)
    {
        (*this)(name, field);
    }

    // After visit: every key of the object must have been a field
    void finish()
    {
        if (!ok()) return;
        for (auto& kv : frames.back().object->object_items())
            if (!frames.back().used.count(kv.first))
            {
                std::string valid;
                for (const char* n : frames.back().names) valid += std::string(valid.empty() ? "" : ", ") + n;
                return fail(path + kv.first, "unknown field (fields: " + valid + ")");
            }
    }

    template <class T>
    void read(const Json& j, T& v, const std::string& where)
    {
        using namespace VisitDetail;
        if (!ok()) return;
        if constexpr (std::is_same_v<T, bool>)
        {
            if (!j.is_bool()) return fail(where, "expected a bool");
            if (apply) v = j.bool_value();
        }
        else if constexpr (std::is_enum_v<T>)
        {
            // Only reachable for a component that registration rejects (see JsonWriter::write)
            int64_t value = 0;
            if (!readInteger(j, value, where)) return;
            if (apply) v = T(value);
        }
        else if constexpr (isInteger<T>)
        {
            T value{};
            if (!readInteger(j, value, where)) return;
            if (apply) v = value;
        }
        else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            double d = 0;
            if (j.is_number())
                d = j.number_value();
            else if (j.is_string() && j.string_value() == "nan")
                d = std::numeric_limits<double>::quiet_NaN();
            else if (j.is_string() && j.string_value() == "inf")
                d = std::numeric_limits<double>::infinity();
            else if (j.is_string() && j.string_value() == "-inf")
                d = -std::numeric_limits<double>::infinity();
            else
                return fail(where, "expected a number");
            // Writes from agents (finiteOnly): no nan or inf (also not from an overflowing literal such as 1e400)
            if (finiteOnly && !std::isfinite(d)) return fail(where, "must be a finite number");
            if constexpr (std::is_same_v<T, float>)
                if (std::isfinite(d) && std::abs(d) > double(std::numeric_limits<float>::max()))
                    return fail(where, "out of the float range");
            if (apply) v = T(d);
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            if (!j.is_string()) return fail(where, "expected a string");
            if (apply) v = j.string_value();
        }
        else if constexpr (std::is_same_v<T, Entity>)
        {
            Entity e;
            if (!parseEntity(j, e, where)) return;
            if (apply) v = e;
        }
        else if constexpr (std::is_same_v<T, PersistentId>)
        {
            uint64_t value = 0;
            if (!readInteger(j, value, where)) return;
            if (apply) v.value = value;
        }
        else if constexpr (isDoubleMath<T> || isFloatMath<T>)
        {
            constexpr int n = int(T::RowsAtCompileTime * T::ColsAtCompileTime);
            if (!j.is_array() || int(j.array_items().size()) != n)
                return fail(where, "expected an array of " + std::to_string(n) + " numbers");
            int k = 0;
            for (int c = 0; c < int(T::ColsAtCompileTime); ++c)
                for (int r = 0; r < int(T::RowsAtCompileTime); ++r, ++k)
                {
                    auto value = v(r, c);
                    read(j.array_items()[size_t(k)], value, where + "[" + std::to_string(k) + "]");
                    if (apply) v(r, c) = value;
                }
        }
        else if constexpr (isQuat<T>)
        {
            if (!j.is_array() || j.array_items().size() != 4) return fail(where, "expected [w, x, y, z]");
            // Read into locals in both modes (the dry run checks the length too); normalized when written
            double c[4]         = {0, 0, 0, 0};
            const bool wasApply = apply;
            apply               = true;
            for (size_t k = 0; k < 4; ++k) read(j.array_items()[k], c[k], where + "[" + std::to_string(k) + "]");
            apply = wasApply;
            if (!ok()) return;
            // Writes from agents (finiteOnly): a rotation, normalized; unit quaternions stay bit-exact (round trips)
            // The length scaled by the largest coefficient (no overflow for huge values)
            const double m =
                std::max(std::max(std::abs(c[0]), std::abs(c[1])), std::max(std::abs(c[2]), std::abs(c[3])));
            double s[4] = {0, 0, 0, 0};
            double ns   = 0;  // length of c / m
            if (m > 0 && std::isfinite(m))
            {
                for (int k = 0; k < 4; ++k) s[k] = c[k] / m;
                ns = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3]);
            }
            const double n  = m * ns;
            const bool unit = std::abs(n - 1) <= 1e-12;
            if (finiteOnly && !(n >= 1e-12)) return fail(where, "a rotation needs a quaternion of length > 0");
            if (apply)
            {
                if (finiteOnly && !unit)
                    v = T(s[0] / ns, s[1] / ns, s[2] / ns, s[3] / ns);
                else
                    v = T(c[0], c[1], c[2], c[3]);
            }
        }
        else if constexpr (IsStdVector<T>::value)
        {
            if (!j.is_array()) return fail(where, "expected an array");
            // The whole vector is replaced; elements start from their default (a dry run reads into a scratch copy)
            T scratch(j.array_items().size());
            const bool wasApply = apply;
            for (size_t k = 0; k < scratch.size(); ++k)
            {
                typename T::value_type element{};
                apply = true;  // elements are read into the scratch vector in both modes
                read(j.array_items()[k], element, where + "[" + std::to_string(k) + "]");
                apply      = wasApply;
                scratch[k] = std::move(element);
            }
            apply = wasApply;
            if (apply && ok()) v = std::move(scratch);
        }
        else if constexpr (IsStdArray<T>::value)
        {
            if (!j.is_array() || j.array_items().size() != v.size())
                return fail(where, "expected an array of " + std::to_string(v.size()) + " elements");
            for (size_t k = 0; k < v.size(); ++k) read(j.array_items()[k], v[k], where + "[" + std::to_string(k) + "]");
        }
        else if constexpr (IsVisitStruct<T>::value)
        {
            if (!j.is_object()) return fail(where, "expected an object");
            if (frames.size() >= kMaxStructDepth) return fail(where, "nested too deep");
            const std::string saved = path;
            push(j, where + ".");
            v.visit(*this);
            finish();
            frames.pop_back();
            path = saved;
        }
        else
        {
            static_assert(AlwaysFalse<T>::value, "JsonReader: unsupported field type (see Visit.h)");
        }
    }

    // {"index": i, "generation": g} or "i:g"
    bool parseEntity(const Json& j, Entity& e, const std::string& where)
    {
        uint32_t index = 0, generation = 0;
        if (j.is_object())
        {
            for (auto& kv : j.object_items())
                if (kv.first != "index" && kv.first != "generation")
                {
                    fail(where + "." + kv.first, "unknown field of an entity");
                    return false;
                }
            if (!readInteger(j["index"], index, where + ".index") ||
                !readInteger(j["generation"], generation, where + ".generation"))
                return false;
        }
        else if (j.is_string())
        {
            const std::string& s = j.string_value();
            const size_t colon   = s.find(':');
            if (colon == std::string::npos || !parseUnsigned(s.substr(0, colon), index) ||
                !parseUnsigned(s.substr(colon + 1), generation))
            {
                fail(where, "expected an entity \"index:generation\"");
                return false;
            }
        }
        else
        {
            fail(where, "expected an entity {\"index\", \"generation\"} or \"index:generation\"");
            return false;
        }
        e = Entity{index, generation};
        return true;
    }

   private:
    struct Frame
    {
        const Json* object;
        std::set<std::string> used;
        std::vector<const char*> names;  // the fields visited (for the error message)
    };
    static constexpr size_t kMaxStructDepth = 64;

    void push(const Json& object, const std::string& prefix)
    {
        frames.push_back(Frame{&object, {}, {}});
        path = prefix;
        if (frames.size() == 1 && !object.is_object()) fail("", "expected an object");
    }
    const Json* take(const char* name)
    {
        if (!ok()) return nullptr;
        Frame& f = frames.back();
        if (!f.object->is_object()) return nullptr;
        auto& items = f.object->object_items();
        auto it     = items.find(name);
        if (it == items.end()) return nullptr;
        f.used.insert(name);
        return &it->second;
    }
    void fail(const std::string& where, const std::string& message)
    {
        if (error.empty()) error = (where.empty() ? std::string("document") : where) + ": " + message;
    }
    template <class T>
    static bool parseUnsigned(const std::string& s, T& out)
    {
        if (s.empty() || s.size() > 20) return false;
        uint64_t v = 0;
        for (char c : s)
        {
            if (c < '0' || c > '9') return false;
            const uint64_t d = uint64_t(c - '0');
            if (v > (std::numeric_limits<uint64_t>::max() - d) / 10) return false;
            v = v * 10 + d;
        }
        if (v > uint64_t(std::numeric_limits<T>::max())) return false;
        out = T(v);
        return true;
    }
    template <class T>
    bool readInteger(const Json& j, T& out, const std::string& where)
    {
        if (j.is_number())
        {
            const double d = j.number_value();
            if (d != std::floor(d) || !std::isfinite(d))
            {
                fail(where, "expected an integer");
                return false;
            }
            // Range check in double (exact for every bound used here up to 2^53; beyond that strings are exact)
            if (d < double(std::numeric_limits<T>::min()) || d > double(std::numeric_limits<T>::max()))
            {
                fail(where, "out of range");
                return false;
            }
            // A JSON number of 2^53 or more may already be rounded (2^53 + 1 parses as 2^53): strings are exact
            if (std::abs(d) >= 9007199254740992.0)
            {
                fail(where, "integers of 2^53 or more must be strings (exact)");
                return false;
            }
            out = T(d);
            return true;
        }
        if (j.is_string())
        {
            const std::string& s = j.string_value();
            if (std::is_signed_v<T> && !s.empty() && s[0] == '-')
            {
                uint64_t magnitude = 0;
                if (!parseUnsigned(s.substr(1), magnitude) || magnitude > uint64_t(std::numeric_limits<T>::max()) + 1)
                {
                    fail(where, "expected an integer in range");
                    return false;
                }
                out = T(-int64_t(magnitude - 1) - 1);
                return true;
            }
            using U = std::make_unsigned_t<T>;
            U value = 0;
            if (!parseUnsigned(s, value) || uint64_t(value) > uint64_t(std::numeric_limits<T>::max()))
            {
                fail(where, "expected an integer in range");
                return false;
            }
            out = T(value);
            return true;
        }
        fail(where, "expected an integer");
        return false;
    }

    bool apply;
    bool finiteOnly;
    std::vector<Frame> frames;
    std::string path;
    std::string error;
};

// The component (or event) as a JSON object
template <class T>
json11::Json toJson(const T& value)
{
    return JsonWriter::write(value);
}

// Merges the fields present in json into value; false (and value unchanged) on any error. finiteOnly: nan and inf
// are errors (writes from the world API); quaternions are normalized (zero length is an error)
template <class T>
bool fromJson(const json11::Json& json, T& value, std::string* error = nullptr, bool finiteOnly = false)
{
    {
        // Dry run on a default instance: the structure, types and ranges do not depend on the current values
        T scratch{};
        JsonReader check(json, false, finiteOnly);
        scratch.visit(check);
        check.finish();
        if (!check.ok())
        {
            if (error) *error = check.errorMessage();
            return false;
        }
    }
    JsonReader reader(json, true, finiteOnly);
    value.visit(reader);
    reader.finish();
    if (error && !reader.ok()) *error = reader.errorMessage();
    return reader.ok();
}

}  // namespace Engine
}  // namespace Saiga
