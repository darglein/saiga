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

#ifdef WIN32
#    include <malloc.h>
#endif


namespace Saiga
{
// returns true if the pointer is actually aligned to the given size
template <typename T, int alignment>
constexpr bool isAligned(const T* ptr)
{
    return (((uintptr_t)ptr) % (alignment)) == 0;
}

template <typename T>
constexpr bool isAligned(const T* ptr)
{
    return isAligned<T, alignof(T)>(ptr);
}



/**
 * Get an aligned piece of memory.
 * Alignment must be a power of 2!
 */
template <size_t Alignment>
inline void* aligned_malloc(size_t size)
{
    auto num = iAlignUp(size, Alignment);
#ifdef WIN32
    // Windows doesn't implement std::aligned_alloc :(
    return _aligned_malloc(num, Alignment);
#elif defined(IS_CUDA)
    // nvcc currently doesn't support std::aligned_alloc (CUDA 10)
    // we cannot use the trick of padding the beginnning because that would break
    // allocating in a .cpp file and freeing in .cu.
    // So let's just hope malloc is already aligned :(
    auto ptr = std::malloc(num);
    if (!isAligned<void, Alignment>(ptr)) throw std::runtime_error("Malloc Not Aligned!");
    return ptr;
#else
    return std::aligned_alloc(Alignment, num);
#endif
}

inline void aligned_free(void* ptr)
{
#ifdef WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

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
        return static_cast<pointer>(aligned_malloc<Alignment>(num));
        //        num = iAlignUp(num, Alignment);
        //        return static_cast<pointer>(std::aligned_alloc(Alignment, num));
    }

    void deallocate(pointer p, size_type /*num*/) { aligned_free(p); }
};

template <typename T, size_t Alignment = alignof(T)>
using AlignedVector = std::vector<T, aligned_allocator<T, Alignment>>;

// An aligned std::vector, which makes sure that the allocated data matches the alignment of T
// template <typename T>
// using AlignedVector = std::vector<T, aligned_allocator<T, alignof (T)>>;


/**
 *  Basically a copy paste of the gcc make_shared implementation, but with the saiga aligned allocator.
 *
 */
template <typename _Tp, size_t Alignment, typename... _Args>
inline std::shared_ptr<_Tp> make_aligned_shared(_Args&&... __args)
{
    // currently using "allocate_shared" with aligned allocator doesn't work.
    // my guess is that the control block is inserted at the beginning and therefore the actual data is after the
    // control block without alignment
    auto ptr = (_Tp*)aligned_malloc<Alignment>(sizeof(_Tp));
    new (ptr) _Tp(std::forward<_Args>(__args)...);
    //    return std::shared_ptr<_Tp>(ptr, &aligned_free);
    return std::shared_ptr<_Tp>(ptr, [](_Tp* ptr) {
        ptr->~_Tp();
        aligned_free(ptr);
    });

    //    typedef typename std::remove_cv<_Tp>::type _Tp_nc;
    //    return std::allocate_shared<_Tp>(aligned_allocator<_Tp_nc, Alignment>(), std::forward<_Args>(__args)...);
}

template <typename _Tp, typename... _Args>
inline std::shared_ptr<_Tp> make_aligned_shared(_Args&&... __args)
{
    return make_aligned_shared<_Tp, alignof(_Tp), _Args...>(std::forward<_Args>(__args)...);
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

template <typename _Tp, typename... _Args>
inline std::unique_ptr<_Tp> make_aligned_unique(_Args&&... __args)
{
    return make_aligned_unique<_Tp, alignof(_Tp), _Args...>(std::forward<_Args>(__args)...);
}


template <typename T, int alignment>
struct SAIGA_ALIGN(alignment) AlignedStruct
{
    T element;

    T& operator()() { return element; }
};



}  // namespace Saiga
