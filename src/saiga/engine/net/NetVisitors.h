/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The network form of visit structs (docs/engine_design.md §13.4; S16): NetWriter and NetReader. Like the binary
// visitors (VisitBinary.h, little endian), except that Entity fields travel as NetIds (varints; 0: none) through a
// NetIdResolver, and Quantize{q} fields as zigzag varints of round(v / q) (quaternions are normalized after reading).
// NetFlattener lists the leaves of a value for the replica comparison of the host + client scenario variant.

#pragma once

#include "saiga/engine/ecs/VisitBinary.h"
#include "saiga/engine/net/Varint.h"

#include <cmath>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
using NetId = uint32_t;

// Entity <-> NetId of one world (the host's table or a client's map)
class NetIdResolver
{
   public:
    virtual ~NetIdResolver() = default;
    // 0 if the entity is not replicated
    virtual NetId netIdOf(Entity e) const = 0;
    // Entity{} for 0 or an unknown NetId
    virtual Entity entityOfNetId(NetId id) const = 0;
};

namespace NetDetail
{
// A type whose network form equals its binary form (no Entity inside, no visit struct that may quantize): containers
// of it are written and read in one piece
template <class T>
constexpr bool plain()
{
    using namespace VisitDetail;
    if constexpr (IsStdVector<T>::value || IsStdArray<T>::value)
        return plain<typename T::value_type>();
    else
        return std::is_arithmetic_v<T> || std::is_same_v<T, std::string> || std::is_same_v<T, PersistentId> ||
               isDoubleMath<T> || isFloatMath<T> || isQuat<T>;
}

// round(v / q) as an integer, clamped (non-finite values become 0: they cannot be quantized)
inline int64_t quantize(double v, double q)
{
    const double x = v / q;
    if (!std::isfinite(x)) return 0;
    if (x > 4.0e18) return int64_t(4.0e18);
    if (x < -4.0e18) return -int64_t(4.0e18);
    return int64_t(std::llround(x));
}
}  // namespace NetDetail

class NetWriter
{
   public:
    NetWriter(std::vector<uint8_t>& out, const NetIdResolver* resolver) : out(out), resolver(resolver) {}

    bool ok() const { return err.empty(); }
    const std::string& error() const { return err; }

    template <class T>
    void operator()(const char*, T& field)
    {
        write(field);
    }
    template <class E, size_t N>
    void operator()(const char* name, E& field, const EnumNames<N>& names)
    {
        BinaryWriter w(out);
        w(name, field, names);
        if (!w.ok()) setError(w.error());
    }
    template <class T>
    void operator()(const char*, T& field, const Quantize& q)
    {
        writeQuantized(field, q.step);
    }

    template <class T>
    void write(const T& v)
    {
        using namespace VisitDetail;
        if constexpr (std::is_same_v<T, Entity>)
            writeVarint(out, resolver ? resolver->netIdOf(v) : 0);
        else if constexpr ((IsStdVector<T>::value || IsStdArray<T>::value) && NetDetail::plain<T>())
        {
            BinaryWriter w(out);
            w.write(v);
            if (!w.ok()) setError(w.error());
        }
        else if constexpr (IsStdVector<T>::value)
        {
            BinaryWriter w(out);
            w.write(uint64_t(v.size()));
            for (const auto& e : v) write(e);
        }
        else if constexpr (IsStdArray<T>::value)
        {
            for (const auto& e : v) write(e);
        }
        else if constexpr (IsVisitStruct<T>::value)
            const_cast<T&>(v).visit(*this);
        else
        {
            BinaryWriter w(out);
            w.write(v);
            if (!w.ok()) setError(w.error());
        }
    }

   private:
    template <class T>
    void writeQuantized(const T& v, double q)
    {
        using namespace VisitDetail;
        if constexpr (std::is_same_v<T, double>)
            writeVarint(out, zigzag(NetDetail::quantize(v, q)));
        else if constexpr (isQuat<T>)
        {
            writeQuantized(double(v.w()), q);
            writeQuantized(double(v.x()), q);
            writeQuantized(double(v.y()), q);
            writeQuantized(double(v.z()), q);
        }
        else
        {
            for (int c = 0; c < int(T::ColsAtCompileTime); ++c)
                for (int r = 0; r < int(T::RowsAtCompileTime); ++r) writeQuantized(double(v(r, c)), q);
        }
    }
    void setError(const std::string& e)
    {
        if (err.empty()) err = e;
    }

    std::vector<uint8_t>& out;
    const NetIdResolver* resolver;
    std::string err;
};

/**
 * Reads the network form. Input is untrusted: every check runs in every build. An Entity field whose NetId is not
 * known (yet) reads as Entity{} and sets unresolved() (the caller decodes the value again when the target arrives).
 */
class NetReader
{
   public:
    NetReader(const uint8_t* data, size_t size, const NetIdResolver* resolver)
        : data(data), size(size), resolver(resolver)
    {
    }

    bool ok() const { return err.empty(); }
    const std::string& error() const { return err; }
    bool unresolved() const { return missing; }
    size_t position() const { return pos; }
    bool atEnd() const { return pos == size; }

    template <class T>
    void operator()(const char*, T& field)
    {
        read(field);
    }
    template <class E, size_t N>
    void operator()(const char* name, E& field, const EnumNames<N>& names)
    {
        if (!err.empty()) return;
        BinaryReader r(data + pos, size - pos);
        r(name, field, names);
        finishLeaf(r);
    }
    template <class T>
    void operator()(const char*, T& field, const Quantize& q)
    {
        readQuantized(field, q.step);
        if constexpr (VisitDetail::isQuat<T>)
        {
            const double n = std::sqrt(double(field.w()) * field.w() + double(field.x()) * field.x() +
                                       double(field.y()) * field.y() + double(field.z()) * field.z());
            if (n > 0 && std::isfinite(n)) field = T(field.w() / n, field.x() / n, field.y() / n, field.z() / n);
        }
    }

    template <class T>
    void read(T& v)
    {
        using namespace VisitDetail;
        if (!err.empty()) return;
        if constexpr (std::is_same_v<T, Entity>)
        {
            uint64_t id;
            if (!readVarint(data, size, pos, id) || id > 0xFFFFFFFFull)
            {
                fail("a malformed NetId");
                return;
            }
            v = Entity{};
            if (id == 0) return;
            if (resolver) v = resolver->entityOfNetId(NetId(id));
            if (!v.valid()) missing = true;
        }
        else if constexpr ((IsStdVector<T>::value || IsStdArray<T>::value) && NetDetail::plain<T>())
        {
            BinaryReader r(data + pos, size - pos);
            r.read(v);
            finishLeaf(r);
        }
        else if constexpr (IsStdVector<T>::value)
        {
            uint64_t n = 0;
            read(n);
            if (!err.empty()) return;
            // Every element takes at least one byte (a quantized value or an Entity: a varint); elements without bytes
            // are capped like the binary reader
            const uint64_t limit = minEncodedSize<typename T::value_type>() == 0 ? kMaxEmptyElements : (size - pos);
            if (n > limit)
            {
                fail("a vector longer than the input");
                return;
            }
            v.clear();
            v.resize(size_t(n));
            for (auto& e : v) read(e);
        }
        else if constexpr (IsStdArray<T>::value)
        {
            for (auto& e : v) read(e);
        }
        else if constexpr (IsVisitStruct<T>::value)
        {
            if (depth >= kMaxStructDepth)
            {
                fail("nested too deep");
                return;
            }
            depth++;
            v.visit(*this);
            depth--;
        }
        else
        {
            BinaryReader r(data + pos, size - pos);
            r.read(v);
            finishLeaf(r);
        }
    }

   private:
    template <class T>
    void readQuantized(T& v, double q)
    {
        using namespace VisitDetail;
        if constexpr (std::is_same_v<T, double>)
        {
            uint64_t z;
            if (!err.empty()) return;
            if (!readVarint(data, size, pos, z))
            {
                fail("a malformed quantized value");
                return;
            }
            v = double(unzigzag(z)) * q;
        }
        else if constexpr (isQuat<T>)
        {
            double w = 0, x = 0, y = 0, z = 0;
            readQuantized(w, q);
            readQuantized(x, q);
            readQuantized(y, q);
            readQuantized(z, q);
            if (err.empty()) v = T(w, x, y, z);
        }
        else
        {
            for (int c = 0; c < int(T::ColsAtCompileTime); ++c)
                for (int r = 0; r < int(T::RowsAtCompileTime); ++r)
                {
                    double x = 0;
                    readQuantized(x, q);
                    v(r, c) = x;
                }
        }
    }
    void finishLeaf(const BinaryReader& r)
    {
        if (!r.ok())
            fail(r.error());
        else
            pos += r.position();
    }
    void fail(const std::string& message)
    {
        if (err.empty()) err = message + " (at byte " + std::to_string(pos) + ")";
    }

    const uint8_t* data;
    size_t size;
    size_t pos = 0;
    const NetIdResolver* resolver;
    std::string err;
    bool missing = false;
    int depth    = 0;
};

// One leaf of a flattened value: quantized numbers with their step, everything else as its exact network bytes
struct NetLeaf
{
    std::string name;
    double step  = 0;  // > 0: a quantized number (value)
    double value = 0;
    std::vector<uint8_t> exact;
};

// The leaves of a value in visit order (Entity fields as NetIds), for comparing replicas field by field
class NetFlattener
{
   public:
    NetFlattener(std::vector<NetLeaf>& out, const NetIdResolver* resolver) : out(out), resolver(resolver) {}

    template <class T>
    void operator()(const char* name, T& field)
    {
        leaf(name, field);
    }
    template <class E, size_t N>
    void operator()(const char* name, E& field, const EnumNames<N>& names)
    {
        NetLeaf l;
        l.name = path(name);
        BinaryWriter w(l.exact);
        w(name, field, names);
        out.push_back(std::move(l));
    }
    template <class T>
    void operator()(const char* name, T& field, const Quantize& q)
    {
        using namespace VisitDetail;
        if constexpr (std::is_same_v<T, double>)
            out.push_back(NetLeaf{path(name), q.step, field, {}});
        else if constexpr (isQuat<T>)
        {
            const char* parts[4] = {"w", "x", "y", "z"};
            const double v[4]    = {field.w(), field.x(), field.y(), field.z()};
            for (int i = 0; i < 4; ++i) out.push_back(NetLeaf{path(name) + "." + parts[i], q.step, v[i], {}});
        }
        else
        {
            for (int c = 0; c < int(T::ColsAtCompileTime); ++c)
                for (int r = 0; r < int(T::RowsAtCompileTime); ++r)
                    out.push_back(NetLeaf{path(name) + "[" + std::to_string(c * int(T::RowsAtCompileTime) + r) + "]",
                                          q.step,
                                          double(field(r, c)),
                                          {}});
        }
    }

   private:
    std::string path(const char* name) const { return prefix.empty() ? name : prefix + "." + name; }

    template <class T>
    void leaf(const char* name, T& v)
    {
        using namespace VisitDetail;
        if constexpr (IsVisitStruct<T>::value)
        {
            const std::string saved = prefix;
            prefix                  = path(name);
            v.visit(*this);
            prefix = saved;
        }
        else if constexpr (IsStdVector<T>::value || IsStdArray<T>::value)
        {
            NetLeaf count;
            count.name = path(name) + ".size";
            BinaryWriter w(count.exact);
            w.write(uint64_t(v.size()));
            out.push_back(std::move(count));
            size_t i = 0;
            for (auto& e : v)
            {
                const std::string element = std::string(name) + "[" + std::to_string(i++) + "]";
                leaf(element.c_str(), e);
            }
        }
        else
        {
            NetLeaf l;
            l.name = path(name);
            NetWriter w(l.exact, resolver);
            w.write(v);
            out.push_back(std::move(l));
        }
    }

    std::vector<NetLeaf>& out;
    const NetIdResolver* resolver;
    std::string prefix;
};

}  // namespace Engine
}  // namespace Saiga
