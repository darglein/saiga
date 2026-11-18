/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Compile-fail cases of the visit field rules (docs/engine_design.md §7.2, R-MATH-1). Built by ctest with
// FIELD_CASE = 0..9 (tests/CMakeLists.txt): case 0 must compile, every other case must fail with its static_assert
// message (checked by the test's PASS_REGULAR_EXPRESSION).

#include "saiga/engine/ecs/Registry.h"

#include <map>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;

#ifndef FIELD_CASE
#    error "FIELD_CASE must be defined"
#endif

enum class Mode : uint8_t
{
    A,
    Count
};

struct NestedWithFloat
{
    static constexpr const char* name = "NestedWithFloat";
    static constexpr uint32_t version = 1;
    float f                           = 0;
    template <class V>
    void visit(V& v)
    {
        v("f", f);
    }
};

#if FIELD_CASE == 0
using FieldType = double;  // control: compiles
#elif FIELD_CASE == 1
using FieldType = float;  // float without allowFloat
#elif FIELD_CASE == 2
using FieldType = Mode;  // enum without EnumNames
#elif FIELD_CASE == 3
using FieldType = int*;  // raw pointer
#elif FIELD_CASE == 4
using FieldType = std::map<int, int>;  // unsupported container
#elif FIELD_CASE == 5
using FieldType = vec3;  // float vector without allowFloat
#elif FIELD_CASE == 6
using FieldType = std::vector<NestedWithFloat>;  // float in a nested struct inside a container
#elif FIELD_CASE == 7
using FieldType = std::vector<bool>;  // not readable by reference
#elif FIELD_CASE == 8
using FieldType = float;  // allowFloat without the Render prefix (see allowFloat below)
#elif FIELD_CASE == 9
using FieldType = char;  // not a fixed-width integer
#endif

struct Component
{
    static constexpr const char* name = "Component";
    static constexpr ComponentId id   = kFirstGameComponent;
    static constexpr uint32_t version = 1;
#if FIELD_CASE == 8
    static constexpr bool allowFloat = true;
#endif
    FieldType field{};
    template <class V>
    void visit(V& v)
    {
        v("field", field);
    }
};

int main()
{
    Registry r;
    r.registerComponent<Component>();
    return 0;
}
