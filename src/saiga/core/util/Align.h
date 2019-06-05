/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/imath.h"

#include <cstdlib>
#include <memory>
#include <vector>

namespace Saiga
{
/**
 * A simple aligned allocator using the standard library.
 * This file requires C++17.
 *
 * Note:
 * It is very similar to Eigen's aligned allocator (and was originally copy pasted from).
 * The changes are:
 *
 * - A templated align-size parameter
 * - Making use of the C++17 std::aligned_alloc function
 */
template <class T, size_t Alignment>
class aligned_allocator : public std::allocator<T>
{
   public:
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    template <class U>
    struct rebind
    {
        typedef aligned_allocator<U, Alignment> other;
    };

    aligned_allocator() : std::allocator<T>() {}

    aligned_allocator(const aligned_allocator& other) : std::allocator<T>(other) {}

    template <class U>
    aligned_allocator(const aligned_allocator<U, Alignment>& other) : std::allocator<T>(other)
    {
    }

    ~aligned_allocator() {}

    pointer allocate(size_type num, const void* /*hint*/ = 0)
    {
        num = num * sizeof(T);
        num = iAlignUp(num, Alignment);
        return static_cast<pointer>(std::aligned_alloc(Alignment, num));
    }

    void deallocate(pointer p, size_type /*num*/) { std::free(p); }
};

// An Aligned std::vector
template <typename T, size_t Alignment = 16>
using AlignedVector = std::vector<T, aligned_allocator<T, Alignment>>;


/**
 *  Basically a copy paste of the gcc make_shared implementation, but with the saiga aligned allocator.
 *
 */
template <typename _Tp, size_t Alignment, typename... _Args>
inline std::shared_ptr<_Tp> make_aligned_shared(_Args&&... __args)
{
    typedef typename std::remove_cv<_Tp>::type _Tp_nc;
    return std::allocate_shared<_Tp>(aligned_allocator<_Tp_nc, Alignment>(), std::forward<_Args>(__args)...);
}

template <typename _Tp, typename... _Args>
inline std::shared_ptr<_Tp> make_aligned_shared(_Args&&... __args)
{
    return make_aligned_shared<_Tp, 16, _Args...>(std::forward<_Args>(__args)...);
}

/**
 * Just forward here because we assume that the class _Tp has overlaoded the
 * new operator correctly.
 */
template <typename _Tp, size_t Alignment, typename... _Args>
inline auto make_aligned_unique(_Args&&... __args)
{
    return std::make_unique<_Tp>(std::forward<_Args>(__args)...);
}



}  // namespace Saiga
