#pragma once

#include "saiga/export.h"
#include <stdint.h>

namespace Saiga
{
struct SAIGA_CORE_API half
{
    uint16_t h;

    half() {}
    half(float f);
    half(uint16_t i);

    operator float();
};
}  // namespace Saiga