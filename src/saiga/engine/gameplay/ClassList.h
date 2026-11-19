/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/TypeCheck.h"
#include "saiga/export.h"

#include <cstring>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
class GameObject;

/**
 * The registered game object classes (docs/engine_design.md §7.9, §9.2): ID, name and factory per class. IDs come
 * from an enum per module like components (EngineClass 0..63, game classes from 64, contiguous); names are unique.
 * A class: a GameObject subclass without constructor parameters with `static constexpr ClassId id` and
 * `static constexpr const char* name` (R-STATE-1, R-OO-1).
 */
class SAIGA_ENGINE_API ClassList
{
   public:
    struct ClassInfo
    {
        ClassId id                               = 0;
        const char* name                         = nullptr;
        std::unique_ptr<GameObject> (*factory)() = nullptr;
        const std::type_info* type               = nullptr;
    };

    template <class T>
    void add();

    // Engine-internal
    // The class with this ID or name, nullptr if none is registered (the name lookup is for the API/MCP boundary)
    const ClassInfo* find(ClassId id) const
    {
        return id < classes.size() && classes[id].factory ? &classes[id] : nullptr;
    }
    const ClassInfo* find(const std::string& name) const;
    // The registration of T (asserts that T is registered under its ID as this C++ type)
    template <class T>
    const ClassInfo& info() const;
    // Contiguous IDs: engine classes 0..n, game classes 64..m
    void validateRegistration() const;
    size_t size() const;
    const std::vector<ClassInfo>& all() const { return classes; }

   private:
    std::vector<ClassInfo> classes;  // by ClassId (factory null: not registered)
};

template <class T>
void ClassList::add()
{
    static_assert(std::is_base_of_v<GameObject, T>, "a class derives from GameObject");
    static_assert(std::is_same_v<std::remove_cv_t<decltype(T::id)>, ClassId>, "T::id must be a ClassId");
    static_assert(std::is_default_constructible_v<T>, "game object classes have no constructor parameters (R-STATE-1)");
    if (classes.size() <= T::id) classes.resize(size_t(T::id) + 1);
    if (classes[T::id].factory)
        SAIGA_EXIT_ERROR(std::string("ClassList::add: ID ") + std::to_string(int(T::id)) + " of " + T::name +
                         " is already used by " + classes[T::id].name);
    for (auto& c : classes)
        SAIGA_ASSERT(!c.factory || std::strcmp(c.name, T::name) != 0,
                     std::string("ClassList::add: the name ") + T::name + " is already used");
    classes[T::id] =
        ClassInfo{T::id, T::name, []() -> std::unique_ptr<GameObject> { return std::make_unique<T>(); }, &typeid(T)};
}

template <class T>
const ClassList::ClassInfo& ClassList::info() const
{
    const ClassInfo* c = find(T::id);
    SAIGA_ASSERT(c, std::string("class ") + T::name + " is not registered");
    SAIGA_ASSERT(
        c->name == T::name || std::strcmp(c->name, T::name) == 0,
        std::string("class ID ") + std::to_string(int(T::id)) + " is registered as " + c->name + ", not " + T::name);
    SAIGA_ASSERT(isRegisteredType(*c->type, typeid(T)),
                 std::string("class ") + T::name + " is another C++ type than the registered one");
    return *c;
}

}  // namespace Engine
}  // namespace Saiga
