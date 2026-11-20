/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/FieldList.h"
#include "saiga/engine/ecs/Visit.h"

#include <cstdint>
#include <string>
#include <vector>

// Schema descriptions of visit structs (docs/engine_design.md §15.2, S11): one entry per saved type with its name,
// version and field list, plus the nested structs reachable from its fields (collected by type, so an empty vector of
// structs still describes its element type). And the staged values of a load: the decoded components or events of one
// type, applied to the world only after the whole save was checked.

namespace Saiga
{
namespace Engine
{
enum class SchemaKind : uint8_t
{
    Component = 0,
    Struct    = 1,
    Event     = 2,
};

struct SchemaType
{
    SchemaKind kind = SchemaKind::Struct;
    std::string name;
    uint32_t id      = 0;  // component or event ID (0 for structs)
    uint32_t version = 0;
    std::vector<FieldInfo> fields;
};

template <class T>
SchemaType describeType(SchemaKind kind, uint32_t id)
{
    return SchemaType{kind, T::name, id, T::version, describeFields<T>()};
}

// Collects the nested visit structs of a type (by type, recursively; each name once)
class NestedTypeCollector
{
   public:
    explicit NestedTypeCollector(std::vector<SchemaType>& out) : out(out) {}

    template <class T>
    void operator()(const char*, T&)
    {
        walk<T>();
    }
    template <class E, size_t N>
    void operator()(const char*, E&, const EnumNames<N>&)
    {
    }
    template <class T>
    void operator()(const char*, T&, const Quantize&)
    {
        walk<T>();
    }

    template <class T>
    void walk()
    {
        using namespace VisitDetail;
        if constexpr (IsStdVector<T>::value || IsStdArray<T>::value)
            walk<typename T::value_type>();
        else if constexpr (IsVisitStruct<T>::value)
        {
            for (const SchemaType& t : out)
                if (t.kind == SchemaKind::Struct && t.name == T::name) return;
            out.push_back(describeType<T>(SchemaKind::Struct, 0));
            T t{};
            t.visit(*this);
        }
    }

   private:
    std::vector<SchemaType>& out;
};

// The nested structs of T's fields (T itself is not added)
template <class T>
void collectNestedTypes(std::vector<SchemaType>& out)
{
    NestedTypeCollector c(out);
    T t{};
    t.visit(c);
}

// Decoded values of one type, waiting for the commit of a load (typed by the pool or queue that created them)
struct StagedValues
{
    virtual ~StagedValues() = default;
    std::vector<uint32_t> indices;  // entity indices (components); unused for events
};

template <class T>
struct StagedOf final : StagedValues
{
    std::vector<T> values;
};

}  // namespace Engine
}  // namespace Saiga
