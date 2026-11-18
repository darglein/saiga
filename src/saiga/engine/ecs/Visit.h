/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/PersistentId.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <type_traits>

// The field rules of visit (docs/engine_design.md §7.2). A component (or a nested struct) lists every field once:
//
//   template <class V> void visit(V& v) { v("hp", hp); v("state", state, EnumNames{StateNames}); }
//
// Supported field types: bool, fixed-size integers, double, enums (only with EnumNames), std::string, Entity,
// PersistentId, Vec2/3/4, Quat, Mat3/4, nested structs with name, version and visit (e.g. ChunkId), and std::array /
// std::vector of supported types (no enums inside containers). float, vec2/3/4, quat, mat3/4 only in a component with
// `static constexpr bool allowFloat = true` (render components, R-MATH-1). Anything else is a compile error in every
// engine visitor (checkFieldType).

namespace Saiga
{
namespace Engine
{
// Names of the values of an enum that ends with Count: v("state", state, EnumNames{StateNames}). The visitor checks at
// compile time that the number of names equals E::Count.
template <size_t N>
struct EnumNames
{
    const char* const* names;
    static constexpr size_t count = N;
    constexpr EnumNames(const char* const (&n)[N]) : names(n) {}
};

// Network quantization of a field (stored for S16): v("pos", p, Quantize{0.01})
struct Quantize
{
    double step;
};

namespace VisitDetail
{
template <class T>
struct AlwaysFalse : std::false_type
{
};

template <class T>
struct IsStdVector : std::false_type
{
};
template <class T, class A>
struct IsStdVector<std::vector<T, A>> : std::true_type
{
};

template <class T>
struct IsStdArray : std::false_type
{
};
template <class T, size_t N>
struct IsStdArray<std::array<T, N>> : std::true_type
{
};

// Accepts every field (used to detect a member template visit)
struct AnyVisitor
{
    template <class... A>
    void operator()(A&&...)
    {
    }
};

template <class T, class = void>
struct IsVisitStruct : std::false_type
{
};
template <class T>
struct IsVisitStruct<T, std::void_t<decltype(T::name), decltype(T::version),
                                    decltype(std::declval<T&>().visit(std::declval<AnyVisitor&>()))>> : std::true_type
{
};

template <class T>
constexpr bool isDoubleMath = std::is_same_v<T, Vec2> || std::is_same_v<T, Vec3> || std::is_same_v<T, Vec4> ||
                              std::is_same_v<T, Mat3> || std::is_same_v<T, Mat4>;
template <class T>
constexpr bool isFloatMath = std::is_same_v<T, vec2> || std::is_same_v<T, vec3> || std::is_same_v<T, vec4> ||
                             std::is_same_v<T, mat3> || std::is_same_v<T, mat4>;
template <class T>
constexpr bool isQuat = std::is_same_v<T, Quat> || std::is_same_v<T, quat>;

template <class T>
constexpr bool isInteger = std::is_integral_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, char> &&
                           !std::is_same_v<T, wchar_t> && !std::is_same_v<T, char16_t> && !std::is_same_v<T, char32_t>;
}  // namespace VisitDetail

// Fails to compile for a field type that visit does not support. AllowFloat: the component declares allowFloat.
template <class T, bool AllowFloat>
constexpr void checkFieldType()
{
    using namespace VisitDetail;
    if constexpr (std::is_same_v<T, bool> || isInteger<T> || std::is_same_v<T, double>)
    {
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        static_assert(AllowFloat, "visit: float fields only in components with allowFloat (render data, R-MATH-1)");
    }
    else if constexpr (std::is_enum_v<T>)
    {
        static_assert(AlwaysFalse<T>::value, "visit: enum fields need EnumNames (and are not allowed in containers)");
    }
    else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, Entity> || std::is_same_v<T, PersistentId> ||
                       isDoubleMath<T> || std::is_same_v<T, Quat>)
    {
    }
    else if constexpr (isFloatMath<T> || std::is_same_v<T, quat>)
    {
        static_assert(AllowFloat, "visit: vec/quat/mat (float) only in components with allowFloat (R-MATH-1)");
    }
    else if constexpr (IsStdVector<T>::value || IsStdArray<T>::value)
    {
        checkFieldType<typename T::value_type, AllowFloat>();
    }
    else if constexpr (IsVisitStruct<T>::value)
    {
        // The fields of the nested struct are checked when a visitor enters it
    }
    else
    {
        static_assert(
            AlwaysFalse<T>::value,
            "visit: unsupported field type (see Visit.h: integers, double, bool, enums with EnumNames, "
            "std::string, Entity, PersistentId, Vec/Quat/Mat, nested visit structs, std::array, std::vector)");
    }
}

template <class T, class = void>
struct AllowsFloat : std::false_type
{
};
template <class T>
struct AllowsFloat<T, std::enable_if_t<T::allowFloat>> : std::true_type
{
};

// The check of a whole component (or nested struct) at compile time: registerComponent instantiates it.
template <bool AllowFloat>
struct FieldTypeCheckVisitor
{
    template <class T>
    void operator()(const char*, T& field)
    {
        checkFieldType<T, AllowFloat>();
        if constexpr (VisitDetail::IsVisitStruct<T>::value) field.visit(*this);
        if constexpr (VisitDetail::IsStdVector<T>::value || VisitDetail::IsStdArray<T>::value)
            enterElement<typename T::value_type>();
    }
    template <class E, size_t N>
    void operator()(const char*, E&, const EnumNames<N>&)
    {
        static_assert(std::is_enum_v<E>, "visit: EnumNames on a non-enum field");
        static_assert(N == size_t(E::Count), "visit: the number of EnumNames differs from E::Count");
    }
    template <class T>
    void operator()(const char* name, T& field, const Quantize&)
    {
        static_assert(std::is_same_v<T, double> || VisitDetail::isDoubleMath<T> || std::is_same_v<T, Quat>,
                      "visit: Quantize only on double, Vec/Mat and Quat fields");
        (*this)(name, field);
    }

   private:
    template <class E>
    void enterElement()
    {
        if constexpr (VisitDetail::IsVisitStruct<E>::value)
        {
            E e{};
            e.visit(*this);
        }
        else if constexpr (VisitDetail::IsStdVector<E>::value || VisitDetail::IsStdArray<E>::value)
        {
            enterElement<typename E::value_type>();
        }
    }
};

template <class T>
void checkVisitFields()
{
    T t{};
    FieldTypeCheckVisitor<AllowsFloat<T>::value> check;
    t.visit(check);
}

}  // namespace Engine
}  // namespace Saiga
