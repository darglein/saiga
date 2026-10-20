#include "half.h"

#include <cstring>
#include <stdint.h>


namespace Saiga
{
static inline uint32_t uint32_nor(uint32_t a, uint32_t b)
{
    return ~(a | b);
}

static inline uint32_t uint32_andc(uint32_t a, uint32_t b)
{
    return (a & (~b));
}

static inline uint32_t uint32_and(uint32_t a, uint32_t b)
{
    return (a & b);
}

static inline uint32_t uint32_or(uint32_t a, uint32_t b)
{
    return (a | b);
}

static inline uint32_t uint32_or3(uint32_t a, uint32_t b, uint32_t c)
{
    return (a | b | c);
}

static inline uint32_t uint32_mux(uint32_t mask, uint32_t a, uint32_t b)
{
    return ((mask & (uint32_t)a) | ((~mask) & (uint32_t)b));
}

// static inline uint32_t uint32_gte(uint32_t a, uint32_t b)
// {
//     return ~uint32_lt(a, b);
// }
//
// static inline uint32_t uint32_gt(uint32_t a, uint32_t b)
// {
//     // NOTE: Result is invalid when b=INT32_MIN, a=INT32_MAX
//     // For the purposes used in half.c the result is always valid
//     return (uint32_t)((int32_t)(b - a) >> 31);
// }

static inline uint32_t uint32_nez(uint32_t a)
{
    return (uint32_t)((int32_t)(a | -a) >> 31);
}

// static inline uint32_t uint32_nez_p(uint32_t a)
// {
//     return (uint32_t)((uint32_t)(a | -a) >> 31);
// }

static inline uint32_t uint32_eq(uint32_t a, uint32_t b)
{
    return (~uint32_nez(a - b));
}

static inline uint32_t uint32_sll(uint32_t a, uint32_t sa)
{
    return (a << sa);
}

static inline uint32_t uint32_cp(uint32_t a)
{
    return (a);
}

static inline uint32_t uint32_add(uint32_t a, uint32_t b)
{
    return (a + b);
}

static inline uint32_t uint32_sub(uint32_t a, uint32_t b)
{
    return (a - b);
}

// static inline uint16_t uint16_nez_p(uint16_t a)
// {
//     return (uint16_t)((uint16_t)(a | -a) >> 15);
// }
//
// static inline uint16_t uint16_eqz_p(uint16_t a)
// {
//     return ~uint16_nez_p(a);
// }

static inline uint16_t uint16_and(uint16_t a, uint16_t b)
{
    return (a & b);
}

static inline uint16_t uint16_or(uint16_t a, uint16_t b)
{
    return (a | b);
}

static inline uint16_t uint16_add(uint16_t a, uint16_t b)
{
    return (a + b);
}

static inline uint16_t uint16_addm(uint16_t a, uint16_t b, uint16_t mask)
{
    return ((a + b) & mask);
}

static inline uint16_t uint16_sub(uint16_t a, uint16_t b)
{
    return (a - b);
}

static inline uint16_t uint16_srl(uint16_t a, uint16_t sa)
{
    return (a >> sa);
}

static inline uint16_t uint16_srlm(uint16_t a, uint16_t sa, uint16_t mask)
{
    return ((a >> sa) & mask);
}

static inline uint16_t uint16_not(uint16_t a)
{
    return (~a);
}

static inline uint16_t uint16_cntlz(uint16_t x)
{
    const uint16_t x0  = uint16_srl(x, 1);
    const uint16_t x1  = uint16_or(x, x0);
    const uint16_t x2  = uint16_srl(x1, 2);
    const uint16_t x3  = uint16_or(x1, x2);
    const uint16_t x4  = uint16_srl(x3, 4);
    const uint16_t x5  = uint16_or(x3, x4);
    const uint16_t x6  = uint16_srl(x5, 8);
    const uint16_t x7  = uint16_or(x5, x6);
    const uint16_t x8  = uint16_not(x7);
    const uint16_t x9  = uint16_srlm(x8, 1, 0x5555);
    const uint16_t xA  = uint16_sub(x8, x9);
    const uint16_t xB  = uint16_and(xA, 0x3333);
    const uint16_t xC  = uint16_srlm(xA, 2, 0x3333);
    const uint16_t xD  = uint16_add(xB, xC);
    const uint16_t xE  = uint16_srl(xD, 4);
    const uint16_t xF  = uint16_addm(xD, xE, 0x0f0f);
    const uint16_t x10 = uint16_srl(xF, 8);
    const uint16_t x11 = uint16_addm(xF, x10, 0x001f);

    return (x11);
}

uint16_t half_from_float(uint32_t f)
{
    // Round to nearest, ties to even (the IEEE default and what the hardware conversions do).
    const uint16_t sign = (f >> 16) & 0x8000;
    const uint32_t a    = f & 0x7fffffff;

    if (a >= 0x7f800000)
    {
        // Inf stays inf. NaN stays NaN: keep the upper payload bits and set the quiet bit, so that a NaN whose
        // payload is only in the low bits does not turn into inf.
        return a == 0x7f800000 ? (sign | 0x7c00) : (sign | 0x7e00 | ((a >> 13) & 0x3ff));
    }
    if (a >= 0x477ff000)
    {
        // >= 65520 (halfway between the largest half 65504 and 65536) rounds to inf.
        return sign | 0x7c00;
    }
    if (a >= 0x38800000)
    {
        // Normal half: rebias the exponent (127 -> 15). A carry of the rounding increments the exponent.
        uint32_t h   = (a - 0x38000000) >> 13;
        uint32_t rem = a & 0x1fff;
        if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
        return sign | (uint16_t)h;
    }
    if (a <= 0x33000000)
    {
        // <= 2^-25 (half of the smallest denormal 2^-24) rounds to zero.
        return sign;
    }
    // Denormal half: value = m * 2^(e - 150), the result is value / 2^-24 = m >> (126 - e) with shift in [14, 24].
    const uint32_t e     = a >> 23;
    const uint32_t m     = (a & 0x7fffff) | 0x800000;
    const uint32_t shift = 126 - e;
    uint32_t h           = m >> shift;
    const uint32_t rem   = m & ((1u << shift) - 1);
    const uint32_t mid   = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (h & 1))) h++;
    return sign | (uint16_t)h;
}

uint32_t half_to_float(uint16_t h)
{
    const uint16_t one                  = 0x0001;
    const uint16_t two                  = 0x0002;
    const uint32_t f_e_mask             = 0x7f800000;
    const uint32_t f_m_mask             = 0x007fffff;
    const uint16_t f_e_bias             = 0x007f;
    const uint16_t h_e_bias             = 0x000f;
    const uint16_t h_s_mask             = 0x8000;
    const uint16_t h_e_mask             = 0x7c00;
    const uint16_t h_m_mask             = 0x03ff;
    const uint16_t h_e_pos              = 0x000a;
    const uint16_t f_e_pos              = 0x0017;
    const uint16_t f_s_pos              = 0x001f;
    const uint16_t h_s_pos              = 0x000f;
    const uint16_t h_f_denorm_sa_offset = 0x000a;
    const uint32_t h_f_bias_offset      = uint16_sub(h_e_bias, f_e_bias);
    const uint16_t h_f_s_pos_offset     = uint16_sub(f_s_pos, h_s_pos);
    const uint16_t h_f_m_pos_offset     = uint16_sub(f_e_pos, h_e_pos);
    const uint32_t h_s                  = uint32_and(h, h_s_mask);
    const uint32_t h_e                  = uint32_and(h, h_e_mask);
    const uint16_t h_m                  = uint32_and(h, h_m_mask);
    const uint32_t f_s                  = uint32_sll(h_s, h_f_s_pos_offset);
    const uint16_t h_e_half_bias        = uint16_srl(h_e, h_e_pos);
    const uint32_t f_e                  = uint16_sub(h_e_half_bias, h_f_bias_offset);
    const uint32_t f_m                  = uint32_sll(h_m, h_f_m_pos_offset);
    const uint32_t is_e_nezero          = uint32_nez(h_e);
    const uint32_t is_m_nezero          = uint32_nez(h_m);
    const uint32_t is_zero              = uint32_nor(is_e_nezero, is_m_nezero);
    const uint32_t is_denorm            = uint32_nor(is_zero, is_e_nezero);
    const uint32_t is_e_flagged         = uint32_eq(h_e, h_e_mask);
    const uint32_t is_inf               = uint32_andc(is_e_flagged, is_m_nezero);
    const uint32_t is_nan               = uint32_and(is_e_flagged, is_m_nezero);
    const uint16_t h_m_nlz              = uint16_cntlz(h_m);
    const uint16_t h_denorm_sa          = uint32_add(h_m_nlz, h_f_denorm_sa_offset);
    const uint32_t h_f_denorm_m_sa      = uint32_sub(h_denorm_sa, two);
    const uint32_t h_f_denorm_m         = uint32_sll(h_m, h_f_denorm_m_sa);
    const uint32_t f_denorm_m           = uint32_and(h_f_denorm_m, f_m_mask);
    const uint32_t h_f_denorm_sa        = uint32_sub(one, h_denorm_sa);
    const uint32_t f_denorm_e           = uint32_add(h_f_denorm_sa, f_e_bias);
    const uint32_t f_denorm_e_packed    = uint32_sll(f_denorm_e, f_e_pos);
    const uint32_t f_denorm             = uint32_or3(f_s, f_denorm_e_packed, f_denorm_m);
    const uint32_t f_inf                = uint32_or(f_s, f_e_mask);
    const uint32_t f_nan                = uint32_or3(f_s, f_e_mask, f_m);
    const uint32_t f_zero               = uint32_cp(f_s);
    const uint32_t f_e_packed           = uint32_sll(f_e, f_e_pos);
    const uint32_t packed               = uint32_or3(f_s, f_e_packed, f_m);
    const uint32_t check_zero_result    = uint32_mux(is_zero, f_zero, packed);
    const uint32_t check_denorm_result  = uint32_mux(is_denorm, f_denorm, check_zero_result);
    const uint32_t check_inf_result     = uint32_mux(is_inf, f_inf, check_denorm_result);
    const uint32_t check_nan_result     = uint32_mux(is_nan, f_nan, check_inf_result);
    const uint32_t result               = check_nan_result;

    return (result);
}

half::half(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    h = half_from_float(bits);
}

half::half(uint16_t i)
{
    h = i;
}

half::operator float() const
{
    uint32_t bits = half_to_float(h);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
}  // namespace Saiga
