/**
 * Copyright (c) 2021 Darius Rückert
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
    using FloatType = typename std::remove_const<T>::type;
    static inline FloatType convert(const T& t) { return FloatType(t); }
    static inline T convertBack(const FloatType& t) { return T(t); }
    static inline T ZeroFloat() { return T(0); }
};

template <>
struct MatchingFloatType<unsigned char>
{
    using FloatType = float;
    static inline FloatType convert(const unsigned char& t)
    {
        return float(t);
    }
    static inline unsigned char convertBack(const FloatType& t)
    {
        return (unsigned char)(round(t));
    }
};


template <>
struct MatchingFloatType<ucvec3>
{
    using FloatType = vec3;
    static inline FloatType convert(const ucvec3& t)
    {
        return t.cast<float>();
    }
    static inline ucvec3 convertBack(const FloatType& t)
    {
        return t.array().round().cast<unsigned char>();
    }
    static inline FloatType ZeroFloat() { return FloatType::Zero(); }
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
        return t.array().round().cast<unsigned char>();
#else
        return FloatType(t);
#endif
    }
    static inline FloatType ZeroFloat() { return FloatType::Zero(); }
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
        return t.array().round().cast<unsigned short>();
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
        if (normalize) f *= NS::scale;
        auto t = Converter::convertBack(f);
        return t;
    }
};



}  // namespace Saiga
