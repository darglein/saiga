/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/observer.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
void Subject::notify()
{
    for (Observer*& view : views) view->notify();
}

}  // namespace Saiga
