/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/ecs/Visit.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <tuple>

// Binary form of visit structs (docs/engine_design.md §7.2): little-endian integers, doubles and floats as their bit
// patterns, bool as one byte, enums as int64, strings and vectors with a u64 length first, nested structs field by
// field, math types element by element (column-major), quaternions as w, x, y, z. No field names and no schema: the
// base for world.hash() (S08) and the save format (S11, which adds the schema tables). The reader validates its input
// in every build (fromBytes returns false with a message).

namespace Saiga
{
namespace Engine
{
// The largest count of a vector whose elements encode to zero bytes (empty structs): written and read as a count only
constexpr uint64_t kMaxEmptyElements = 1 << 16;
// The largest nesting of structs (self-containing structs such as trees), for writing and reading
constexpr int kMaxStructDepth = 64;

template <class T>
size_t minEncodedSize();

// Writes the binary form. Values the reader would reject (more than kMaxEmptyElements zero-byte elements, structs
// nested deeper than kMaxStructDepth, enum values without a name) set an error in every build; the output is then
// incomplete (ok() / error()).
class BinaryWriter
{
   public:
    explicit BinaryWriter(std::vector<uint8_t>& out) : out(out) {}

    bool ok() const { return err.empty(); }
    const std::string& error() const { return err; }

    template <class T>
    void operator()(const char*, T& field)
    {
        write(field);
    }
    template <class E, size_t N>
    void operator()(const char*, E& field, const EnumNames<N>&)
    {
        // The reader rejects values without a name (e.g. E::Count): never write them
        int64_t v = int64_t(field);
        if (v < 0 || uint64_t(v) >= N)
        {
            setError("enum value " + std::to_string(v) + " has no name (" + std::to_string(N) + " names)");
            return;
        }
        writeUnsigned(uint64_t(v), 8);
    }
    template <class T>
    void operator()(const char*, T& field, const Quantize&)
    {
        write(field);
    }

    template <class T>
    void write(const T& v)
    {
        using namespace VisitDetail;
        if constexpr (std::is_same_v<T, bool>)
            writeUnsigned(v ? 1 : 0, 1);
        else if constexpr (isInteger<T>)
            writeUnsigned(uint64_t(v), sizeof(T));
        else if constexpr (std::is_enum_v<T>)
        {
            // Only reachable for a component that registration rejects (enums need EnumNames, checkFieldType); encoded
            // like a named enum so the type check reports the real problem (MSVC stops at the first static_assert)
            writeUnsigned(uint64_t(int64_t(v)), 8);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            uint64_t bits;
            std::memcpy(&bits, &v, 8);
            writeUnsigned(bits, 8);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            writeUnsigned(bits, 4);
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            writeUnsigned(uint64_t(v.size()), 8);
            out.insert(out.end(), v.begin(), v.end());
        }
        else if constexpr (std::is_same_v<T, Entity>)
        {
            writeUnsigned(v.index, 4);
            writeUnsigned(v.generation, 4);
        }
        else if constexpr (std::is_same_v<T, PersistentId>)
            writeUnsigned(v.value, 8);
        else if constexpr (isDoubleMath<T> || isFloatMath<T>)
        {
            for (int c = 0; c < int(T::ColsAtCompileTime); ++c)
                for (int r = 0; r < int(T::RowsAtCompileTime); ++r) write(v(r, c));
        }
        else if constexpr (isQuat<T>)
        {
            write(v.w());
            write(v.x());
            write(v.y());
            write(v.z());
        }
        else if constexpr (IsStdVector<T>::value)
        {
            // The reader accepts at most kMaxEmptyElements elements that encode to zero bytes: never write more (an
            // error in every build, not an assert: the buffer would be unreadable)
            if (v.size() > kMaxEmptyElements && minEncodedSize<typename T::value_type>() == 0)
            {
                setError("vector of " + std::to_string(v.size()) +
                         " elements without bytes (more than kMaxEmptyElements, the reader rejects it)");
                return;
            }
            writeUnsigned(uint64_t(v.size()), 8);
            for (const auto& e : v) write(e);
        }
        else if constexpr (IsStdArray<T>::value)
        {
            for (const auto& e : v) write(e);
        }
        else if constexpr (IsVisitStruct<T>::value)
        {
            // The reader's nesting limit applies to writing too (no buffer that cannot be read back)
            if (depth >= kMaxStructDepth)
            {
                setError("nested deeper than " + std::to_string(kMaxStructDepth) + " structs");
                return;
            }
            depth++;
            // visit is non-const by design (one function for reading and writing)
            const_cast<T&>(v).visit(*this);
            depth--;
        }
        else
        {
            static_assert(AlwaysFalse<T>::value, "BinaryWriter: unsupported field type (see Visit.h)");
        }
    }

   private:
    void writeUnsigned(uint64_t v, int bytes)
    {
        for (int i = 0; i < bytes; ++i) out.push_back(uint8_t(v >> (8 * i)));
    }

    void setError(const std::string& message)
    {
        if (err.empty()) err = message;
    }

    std::vector<uint8_t>& out;
    std::string err;
    int depth = 0;
};

// Minimum number of bytes of one encoded value (bounds the allocation of a vector from its length field). Nested
// structs are summed over their fields at run time.
struct MinSizeVisitor
{
    size_t sum = 0;
    template <class T>
    void operator()(const char*, T&)
    {
        sum += minEncodedSize<T>();
    }
    template <class E, size_t N>
    void operator()(const char*, E&, const EnumNames<N>&)
    {
        sum += 8;
    }
    template <class T>
    void operator()(const char*, T&, const Quantize&)
    {
        sum += minEncodedSize<T>();
    }
};

template <class T>
size_t minEncodedSize()
{
    using namespace VisitDetail;
    if constexpr (std::is_same_v<T, bool>)
        return 1;
    else if constexpr (isInteger<T> || std::is_same_v<T, double> || std::is_same_v<T, float>)
        return sizeof(T);
    else if constexpr (std::is_enum_v<T>)
        return 8;
    else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, Entity> || std::is_same_v<T, PersistentId> ||
                       IsStdVector<T>::value)
        return 8;
    else if constexpr (isDoubleMath<T> || isFloatMath<T>)
        return size_t(T::RowsAtCompileTime) * size_t(T::ColsAtCompileTime) * sizeof(typename T::Scalar);
    else if constexpr (isQuat<T>)
        return 4 * sizeof(typename T::Scalar);
    else if constexpr (IsStdArray<T>::value)
        return std::tuple_size<T>::value * minEncodedSize<typename T::value_type>();
    else
    {
        T t{};
        MinSizeVisitor v;
        t.visit(v);
        return v.sum;
    }
}

/**
 * Reads the binary form. Input is untrusted (saves, network): every check runs in every build (not asserts). On the
 * first error the reader stops, keeps the message and leaves the remaining fields untouched; ok() / error() report it.
 */
class BinaryReader
{
   public:
    BinaryReader(const uint8_t* data, size_t size) : data(data), size(size) {}

    template <class T>
    void operator()(const char*, T& field)
    {
        read(field);
    }
    template <class E, size_t N>
    void operator()(const char*, E& field, const EnumNames<N>&)
    {
        if (failed()) return;
        int64_t v = int64_t(readUnsigned(8));
        if (failed()) return;
        if (v < 0 || uint64_t(v) >= N)
        {
            fail("enum value " + std::to_string(v) + " out of range");
            return;
        }
        field = E(v);
    }
    template <class T>
    void operator()(const char*, T& field, const Quantize&)
    {
        read(field);
    }

    template <class T>
    void read(T& v)
    {
        using namespace VisitDetail;
        if (failed()) return;
        if constexpr (std::is_same_v<T, bool>)
        {
            // Canonical: only 0 and 1 (saves and hashes must not have two encodings of one state)
            uint64_t b = readUnsigned(1);
            if (failed()) return;
            if (b > 1)
            {
                fail("bool byte " + std::to_string(b) + " is neither 0 nor 1");
                return;
            }
            v = b != 0;
        }
        else if constexpr (isInteger<T>)
        {
            uint64_t x = readUnsigned(sizeof(T));
            if (!failed()) v = T(x);
        }
        else if constexpr (std::is_enum_v<T>)
        {
            // See BinaryWriter: only for components that registration rejects
            uint64_t x = readUnsigned(8);
            if (!failed()) v = T(int64_t(x));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            uint64_t bits = readUnsigned(8);
            if (!failed()) std::memcpy(&v, &bits, 8);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            uint32_t bits = uint32_t(readUnsigned(4));
            if (!failed()) std::memcpy(&v, &bits, 4);
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            uint64_t n = readUnsigned(8);
            if (!need(n, "string")) return;
            v.assign(reinterpret_cast<const char*>(data + pos), size_t(n));
            pos += size_t(n);
        }
        else if constexpr (std::is_same_v<T, Entity>)
        {
            uint32_t index      = uint32_t(readUnsigned(4));
            uint32_t generation = uint32_t(readUnsigned(4));
            if (!failed()) v = Entity{index, generation};
        }
        else if constexpr (std::is_same_v<T, PersistentId>)
        {
            uint64_t value = readUnsigned(8);
            if (!failed()) v.value = value;
        }
        else if constexpr (isDoubleMath<T> || isFloatMath<T>)
        {
            for (int c = 0; c < int(T::ColsAtCompileTime); ++c)
                for (int r = 0; r < int(T::RowsAtCompileTime); ++r) read(v(r, c));
        }
        else if constexpr (isQuat<T>)
        {
            read(v.w());
            read(v.x());
            read(v.y());
            read(v.z());
        }
        else if constexpr (IsStdVector<T>::value)
        {
            uint64_t n = readUnsigned(8);
            if (failed()) return;
            // The elements must fit into the remaining input: a corrupt length cannot allocate more than the input
            // allows (each element takes at least minEncodedSize bytes). Elements without bytes (empty structs,
            // zero-length arrays) are written as a count only; their count is capped instead.
            const size_t minSize    = minEncodedSize<typename T::value_type>();
            const uint64_t maxCount = minSize == 0 ? kMaxEmptyElements : (size - pos) / minSize;
            if (n > maxCount)
            {
                fail("vector of " + std::to_string(n) + " elements does not fit into the input");
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
            // Bounded call depth for self-containing structs (a deep tree in untrusted input cannot overflow the stack)
            if (depth >= kMaxDepth)
            {
                fail("nested deeper than " + std::to_string(kMaxDepth) + " structs");
                return;
            }
            depth++;
            v.visit(*this);
            depth--;
        }
        else
        {
            static_assert(AlwaysFalse<T>::value, "BinaryReader: unsupported field type (see Visit.h)");
        }
    }

    // The largest nesting of structs (self-containing structs such as trees)
    static constexpr int kMaxDepth = kMaxStructDepth;

    bool ok() const { return err.empty(); }
    const std::string& error() const { return err; }
    size_t position() const { return pos; }
    bool atEnd() const { return pos == size; }
    size_t remaining() const { return size - pos; }
    // Skips n bytes (false and an error if fewer are left)
    bool skip(uint64_t n)
    {
        if (!need(n, "skipped block")) return false;
        pos += size_t(n);
        return true;
    }

   private:
    bool failed() const { return !err.empty(); }
    void fail(const std::string& message)
    {
        if (err.empty()) err = message + " (at byte " + std::to_string(pos) + " of " + std::to_string(size) + ")";
    }
    bool need(uint64_t n, const char* what)
    {
        if (failed()) return false;
        if (n > size - pos)
        {
            fail(std::string("truncated input: ") + what + " needs " + std::to_string(n) + " bytes");
            return false;
        }
        return true;
    }

    uint64_t readUnsigned(int bytes)
    {
        if (!need(uint64_t(bytes), "value")) return 0;
        uint64_t v = 0;
        for (int i = 0; i < bytes; ++i) v |= uint64_t(data[pos + size_t(i)]) << (8 * i);
        pos += size_t(bytes);
        return v;
    }

    const uint8_t* data;
    size_t size;
    size_t pos = 0;
    int depth  = 0;
    std::string err;
};

// The binary form of a visit struct. Returns false (and *error) for a value the reader would reject; out is then
// incomplete.
template <class T>
bool toBytes(const T& value, std::vector<uint8_t>& out, std::string* error)
{
    // Into a separate buffer: out may be a field of value
    std::vector<uint8_t> bytes;
    BinaryWriter w(bytes);
    w.write(value);
    if (error) *error = w.error();
    out = std::move(bytes);
    return w.ok();
}

// The binary form of a visit struct; stops the program (in every build) for a value the reader would reject
template <class T>
std::vector<uint8_t> toBytes(const T& value)
{
    std::vector<uint8_t> out;
    std::string error;
    if (!toBytes(value, out, &error)) SAIGA_EXIT_ERROR("toBytes: " + error);
    return out;
}

// Reads a visit struct written by toBytes. Returns false on malformed input (truncated, an enum out of range, a vector
// longer than the input, bytes left over) and sets *error if given; value may be partially written then.
template <class T>
bool fromBytes(T& value, const uint8_t* data, size_t size, std::string* error = nullptr)
{
    BinaryReader r(data, size);
    r.read(value);
    std::string message = r.error();
    if (message.empty() && !r.atEnd())
        message = std::to_string(size - r.position()) + " bytes left over after the value";
    if (error) *error = message;
    return message.empty();
}

template <class T>
bool fromBytes(T& value, const std::vector<uint8_t>& bytes, std::string* error = nullptr)
{
    BinaryReader r(bytes.data(), bytes.size());
    r.read(value);
    std::string message = r.error();
    if (message.empty() && !r.atEnd())
        message = std::to_string(bytes.size() - r.position()) + " bytes left over after the value";
    if (error) *error = message;
    return message.empty();
}

}  // namespace Engine
}  // namespace Saiga
