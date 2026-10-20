/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include "imageFormat.h"

#include <climits>
#include <cmath>

#include <type_traits>

namespace Saiga
{
template <typename T>
struct SAIGA_TEMPLATE MatchingFloatType
{
    using NoConstType = typename std::remove_const<T>::type;
    // Integer texels are processed as floating point values. Otherwise sums (for example in copyScaleDownPow2)
    // overflow, and interpolated values are truncated instead of rounded.
    using FloatType =
        typename std::conditional<std::is_integral<NoConstType>::value,
                                  typename std::conditional<(sizeof(NoConstType) <= 2), float, double>::type,
                                  NoConstType>::type;
    HD static inline FloatType convert(const T& t) { return FloatType(t); }
    HD static inline T convertBack(const FloatType& t)
    {
        if constexpr (std::is_integral<NoConstType>::value)
        {
            return T(::round(t));
        }
        else
        {
            return T(t);
        }
    }
    HD static inline FloatType ZeroFloat() { return FloatType(0); }
};

template <>
struct MatchingFloatType<unsigned char>
{
    using FloatType = float;
    HD static inline FloatType convert(const unsigned char& t) { return float(t); }
    HD static inline unsigned char convertBack(const FloatType& t) { return (unsigned char)(round(t)); }
    HD static inline FloatType ZeroFloat() { return 0; }
};


template <>
struct MatchingFloatType<ucvec3>
{
    using FloatType = vec3;
    static inline FloatType convert(const ucvec3& t) { return t.cast<float>(); }
    static inline ucvec3 convertBack(const FloatType& t) { return t.array().round().cast<unsigned char>(); }
    static inline FloatType ZeroFloat() { return FloatType::Zero(); }
};
template <>
struct MatchingFloatType<ucvec4>
{
    using FloatType = vec4;
    static inline FloatType convert(const ucvec4& t) { return t.cast<float>(); }
    static inline ucvec4 convertBack(const FloatType& t) { return t.array().round().cast<unsigned char>(); }
    static inline FloatType ZeroFloat() { return FloatType::Zero(); }
};


template <>
struct MatchingFloatType<usvec3>
{
    using FloatType = vec3;
    static inline FloatType convert(const usvec3& t) { return t.cast<float>(); }
    static inline usvec3 convertBack(const FloatType& t) { return t.array().round().cast<unsigned short>(); }
    static inline FloatType ZeroFloat() { return FloatType::Zero(); }
};

template <typename T, typename ST = float>
struct SAIGA_TEMPLATE NormalizeScale
{
    constexpr static ST scale = ST(1);
};

template <typename ST>
struct NormalizeScale<char, ST>
{
    // char is signed on most platforms, so its largest value is 127 (CHAR_MAX), not 255.
    constexpr static ST scale = ST(CHAR_MAX);
};
template <typename ST>
struct NormalizeScale<unsigned char, ST>
{
    constexpr static ST scale = ST(0xFF);
};
template <typename ST>
struct NormalizeScale<unsigned short, ST>
{
    constexpr static ST scale = ST(0xFFFF);
};



template <typename T, bool normalize = false>
struct SAIGA_TEMPLATE TexelFloatConverter
{
    using ITT       = ImageTypeTemplate<T>;
    using TexelType = T;
    using NS        = NormalizeScale<typename ITT::ChannelType>;
    using Converter = MatchingFloatType<T>;
    using FloatType = typename MatchingFloatType<T>::FloatType;


    // HD, because ImageView::inter uses it and is called in CUDA kernels (for example scaleUp2Linear).
    HD FloatType toFloat(TexelType t)
    {
        auto f = Converter::convert(t);
        return normalize ? f * (1.0f / NS::scale) : f;
    }

    HD TexelType fromFloat(FloatType f)
    {
        if (normalize) f *= NS::scale;
        auto t = Converter::convertBack(f);
        return t;
    }
};



}  // namespace Saiga
