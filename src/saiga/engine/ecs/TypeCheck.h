/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstring>
#include <typeinfo>

namespace Saiga
{
namespace Engine
{
/**
 * Whether `t` is the registered C++ type (component pools and event queues check it before every downcast, so two
 * types with the same ID, name and size are told apart). std::type_info equality works across the engine DLL (the
 * standard libraries compare the mangled names there). Types in unnamed namespaces of different translation units can
 * compare equal, though (Clang with libstdc++ compares their names, which are the same), so for them the type_info
 * object must also be the same one: such a type exists only in its own translation unit, where typeid always yields
 * the same object. On the Itanium ABI (GCC, Clang) their names contain _GLOBAL__N_ (reserved, no user name contains
 * it); MSVC gives each unnamed namespace a unique decorated name.
 */
inline bool isRegisteredType(const std::type_info& registered, const std::type_info& t)
{
    if (!(registered == t)) return false;
#ifndef _MSC_VER
    if (std::strstr(t.name(), "_GLOBAL__N_")) return &registered == &t;
#endif
    return true;
}

}  // namespace Engine
}  // namespace Saiga
