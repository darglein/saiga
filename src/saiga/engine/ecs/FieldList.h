/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Visit.h"

#include <string>
#include <vector>

// The field list of a visit struct: name and type of each field in visit order (docs/engine_design.md §7.2). The
// schema tables of saves (S11) and the inspect tool (S14) are built from it.

namespace Saiga
{
namespace Engine
{
struct FieldInfo
{
    std::string name;
    std::string type;
    // For type "enum": the value names in value order (the name tables of the schema, §15.2)
    std::vector<std::string> enumValues;
    // The network quantization step (Quantize, §13.4; 0: none). Not part of the save schema.
    double quantize = 0;
};

template <class T>
std::string fieldTypeName()
{
    using namespace VisitDetail;
    if constexpr (std::is_same_v<T, bool>)
        return "bool";
    else if constexpr (std::is_enum_v<T>)
        return "enum";  // only reachable for a component that registration rejects (enums need EnumNames)
    else if constexpr (isInteger<T>)
        return std::string(std::is_signed_v<T> ? "int" : "uint") + std::to_string(8 * sizeof(T));
    else if constexpr (std::is_same_v<T, double>)
        return "double";
    else if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_same_v<T, std::string>)
        return "string";
    else if constexpr (std::is_same_v<T, Entity>)
        return "Entity";
    else if constexpr (std::is_same_v<T, PersistentId>)
        return "PersistentId";
    else if constexpr (isDoubleMath<T> || isFloatMath<T>)
    {
        std::string s = T::ColsAtCompileTime == 1 ? "vec" + std::to_string(int(T::RowsAtCompileTime))
                                                  : "mat" + std::to_string(int(T::RowsAtCompileTime));
        return s + (isDoubleMath<T> ? "d" : "f");
    }
    else if constexpr (std::is_same_v<T, Quat>)
        return "quatd";
    else if constexpr (std::is_same_v<T, quat>)
        return "quatf";
    else if constexpr (IsStdVector<T>::value)
        return "vector<" + fieldTypeName<typename T::value_type>() + ">";
    else if constexpr (IsStdArray<T>::value)
        return "array<" + fieldTypeName<typename T::value_type>() + "," + std::to_string(std::tuple_size<T>::value) +
               ">";
    else if constexpr (IsVisitStruct<T>::value)
        return T::name;
    else
        static_assert(AlwaysFalse<T>::value, "fieldTypeName: unsupported field type (see Visit.h)");
}

class FieldListVisitor
{
   public:
    template <class T>
    void operator()(const char* name, T&)
    {
        fields.push_back({name, fieldTypeName<T>()});
    }
    template <class E, size_t N>
    void operator()(const char* name, E&, const EnumNames<N>& names)
    {
        FieldInfo f{name, "enum", {}};
        for (size_t i = 0; i < N; ++i) f.enumValues.push_back(names.names[i]);
        fields.push_back(f);
    }
    template <class T>
    void operator()(const char* name, T& field, const Quantize& q)
    {
        (*this)(name, field);
        fields.back().quantize = q.step;
    }

    std::vector<FieldInfo> fields;
};

// The top-level fields of T (nested structs appear with their name as type; describe them separately)
template <class T>
std::vector<FieldInfo> describeFields()
{
    T t{};
    FieldListVisitor v;
    t.visit(v);
    return v.fields;
}

}  // namespace Engine
}  // namespace Saiga
