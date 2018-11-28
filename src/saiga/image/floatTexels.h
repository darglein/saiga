/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/math.h"
#include "saiga/image/imageFormat.h"

namespace Saiga {


template<typename T>
struct SAIGA_TEMPLATE MatchingFloatType{ using FloatType = T; };

template<> struct MatchingFloatType<ucvec3>{using FloatType = vec3; };
template<> struct MatchingFloatType<ucvec4>{using FloatType = vec4; };



template<typename T, typename ST = float>
struct SAIGA_TEMPLATE NormalizeScale{ constexpr static ST scale = ST(1); };

template<typename ST> struct NormalizeScale<char,ST>{ constexpr static ST scale = ST(255); };
template<typename ST> struct NormalizeScale<unsigned char,ST>{ constexpr static ST scale = ST(255); };




template<typename T, bool normalize = false>
struct SAIGA_TEMPLATE TexelFloatConverter
{
    using ITT = ImageTypeTemplate<T>;
    using TexelType = T;
    using NS = NormalizeScale<typename ITT::ChannelType>;
    using FloatType = typename MatchingFloatType<T>::FloatType;


    FloatType toFloat(TexelType t)
    {
        return normalize ? FloatType(t) / NS::scale : FloatType(t);
    }

    TexelType fromFloat(FloatType f)
    {
//        return TexelType(f);
        return normalize ? TexelType(f * NS::scale) : TexelType(f);
    }

};




}
