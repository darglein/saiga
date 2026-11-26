/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// EntityFieldVisitor (S25, docs/engine_design.md §9.6): calls fn(Entity&) for every Entity field of a visit struct in
// visit order (nested structs, std::vector and std::array elements included). The entity groups of unloaded chunks
// remap the Entity fields of their components with it: to group-local handles Entity{local + 1, 0} when written (and
// back to the original values afterwards: the live component is not copied) and to the new handles when reloaded.

#pragma once

#include "saiga/engine/ecs/Visit.h"

#include <functional>

namespace Saiga
{
namespace Engine
{
// Maps an entity handle (the group writer: live -> local; the loader: local -> new)
using EntityRemap = std::function<Entity(Entity)>;

template <class F>
class EntityFieldVisitor
{
   public:
    explicit EntityFieldVisitor(F& fn) : fn(fn) {}

    template <class T>
    void operator()(const char*, T& field)
    {
        apply(field);
    }
    template <class E, size_t N>
    void operator()(const char*, E&, const EnumNames<N>&)
    {
    }
    template <class T>
    void operator()(const char*, T& field, const Quantize&)
    {
        apply(field);
    }

    template <class T>
    void apply(T& v)
    {
        using namespace VisitDetail;
        if constexpr (std::is_same_v<T, Entity>)
            fn(v);
        else if constexpr (IsStdVector<T>::value || IsStdArray<T>::value)
        {
            if constexpr (!IsVectorOfBool<T>::value)
                for (auto& e : v) apply(e);
        }
        else if constexpr (IsVisitStruct<T>::value)
            v.visit(*this);
    }

   private:
    F& fn;
};

// fn(Entity&) for every Entity field of value
template <class T, class F>
void forEachEntityField(T& value, F&& fn)
{
    EntityFieldVisitor<std::remove_reference_t<F>> v(fn);
    value.visit(v);
}

}  // namespace Engine
}  // namespace Saiga
