/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/core/util/math.h"

#include <map>
#include <vector>

namespace Saiga
{
#if defined(SAIGA_DEBUG)
#    define assert_no_alerror() SAIGA_ASSERT(!sound::checkSoundError())
#else
#    define assert_no_alerror() (void)0
#endif


namespace sound
{
SAIGA_GLOBAL extern void initOpenAL();
SAIGA_GLOBAL extern void quitOpenAL();


SAIGA_GLOBAL bool checkSoundError();

SAIGA_GLOBAL extern std::string getALCErrorString(int err);



}  // namespace sound

}  // namespace Saiga
