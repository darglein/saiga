/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/time/timer.h"
#include "saiga/util/statistics.h"

namespace Saiga {


template<typename F, typename ... Ts>
inline void measureFunction(const std::string& name, int its, F f, Ts&... args)
{
    std::vector<float> timings(its);
    for(int i = 0; i < its; ++i){
        float time;
        {
            ScopedTimer<float> tim(time);
            f(args...);
        }
        timings[i] = time;
    }
    std::cout << "> Measured exectuion time of function " << name << " in ms." << std::endl;
    std::cout << Statistics<float>(timings) << std::endl;
}


template<typename F>
inline void measureObject(const std::string& name, int its, F f)
{
    std::vector<float> timings(its);
    for(int i = 0; i < its; ++i){
        float time;
        {
            ScopedTimer<float> tim(time);
            f();
        }
        timings[i] = time;
    }
    std::cout << "> Measured exectuion time of function " << name << " in ms." << std::endl;
    std::cout << Statistics<float>(timings) << std::endl;
}


}
