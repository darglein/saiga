/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include "imageFormat.h"

namespace Saiga
{
template <typename T>
struct SAIGA_TEMPLATE MatchingFloatType
{
    using FloatType = T;
    static inline FloatType convert(const T& t) { return FloatType(t); }
    static inline T convertBack(const FloatType& t) { return T(t); }
};

template <>
struct MatchingFloatType<ucvec3>
{
    using FloatType = vec3;
    static inline FloatType convert(const ucvec3& t)
    {
#ifdef SAIGA_FULL_EIGEN
        return t.cast<float>();
#else
        return FloatType(t);
#endif
    }
    static inline ucvec3 convertBack(const FloatType& t)
    {
#ifdef SAIGA_FULL_EIGEN
        return t.cast<unsigned char>();
#else
        return FloatType(t);
#endif
    }
};
template <>
struct MatchingFloatType<ucvec4>
{
    using FloatType = vec4;
    static inline FloatType convert(const ucvec4& t)
    {
#ifdef SAIGA_FULL_EIGEN
        return t.cast<float>();
#else
        return FloatType(t);
#endif
    }
    static inline ucvec4 convertBack(const FloatType& t)
    {
#ifdef SAIGA_FULL_EIGEN
        return t.cast<unsigned char>();
#else
        return FloatType(t);
#endif
    }
};


template <>
struct MatchingFloatType<usvec3>
{
    using FloatType = vec3;
    static inline FloatType convert(const usvec3& t)
    {
#ifdef SAIGA_FULL_EIGEN
        return t.cast<float>();
#else
        return FloatType(t);
#endif
    }
    static inline usvec3 convertBack(const FloatType& t)
    {
#ifdef SAIGA_FULL_EIGEN
        return t.cast<unsigned short>();
#else
        return FloatType(t);
#endif
    }
};

template <typename T, typename ST = float>
struct SAIGA_TEMPLATE NormalizeScale
{
    constexpr static ST scale = ST(1);
};

template <typename ST>
struct NormalizeScale<char, ST>
{
    constexpr static ST scale = ST(0xFF);
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


    FloatType toFloat(TexelType t)
    {
        auto f = Converter::convert(t);
        return normalize ? f * (1.0f / NS::scale) : f;
    }

    TexelType fromFloat(FloatType f)
    {
        auto t = Converter::convertBack(f);
        if (normalize) t *= NS::scale;
        return t;
    }
};



}  // namespace Saiga
