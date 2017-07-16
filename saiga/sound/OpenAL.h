#pragma once

#include <saiga/util/glm.h>
#include "saiga/util/assert.h"

#include <vector>
#include <map>

namespace Saiga {

#if defined(SAIGA_DEBUG)
    #define assert_no_alerror() SAIGA_ASSERT(!sound::checkSoundError())
#else
    #define assert_no_alerror() (void)0
#endif


namespace sound {



SAIGA_GLOBAL extern void initOpenAL();
SAIGA_GLOBAL extern void quitOpenAL();


SAIGA_GLOBAL  bool checkSoundError();

SAIGA_GLOBAL  extern std::string getALCErrorString(int err);




}

}
