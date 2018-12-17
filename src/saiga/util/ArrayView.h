/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

// Test if saiga was compiled with CUDA and
// the current project has CUDA in the include dir.
// If yes, generate a constructor and additional functions for thrust device vectors.
#ifdef SAIGA_USE_CUDA
#    if __has_include(<thrust/device_vector.h>)
#        include <thrust/device_vector.h>
#        define SAIGA_GENERATE_THRUST_CONSTRUCTOR
#    endif
#endif

#include <cstddef>
#include <cstdint>

#include <type_traits>

namespace Saiga
{
/**
 * Array View to abstract from Container classes such as std::vector or thrust::device_vector.
 * An ArrayView object is lightweight and contains only the data pointer and the number of elements.
 * ArrayViews can be savely passed to CUDA kernels, if the data pointer points to device memory.
 *
 * A common use-case is to use ArrayView arguments for functions that do not need to change
 * the container class.
 *
 * Thanks to Johannes Pieger for the intitial implementation.
 */

template <typename T>
struct SAIGA_TEMPLATE ArrayView
{
    using value_type      = T;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;
    using size_type       = size_t;
    using difference_type = ptrdiff_t;

    HD ArrayView() : data_(nullptr), n(0) {}
    HD ArrayView(T* data_, size_t n) : data_(data_), n(n) {}

    ArrayView(ArrayView<T> const&) = default;
    ArrayView& operator=(ArrayView<T> const&) = default;


#ifdef SAIGA_GENERATE_THRUST_CONSTRUCTOR
    __host__ ArrayView(thrust::device_vector<typename std::remove_const<T>::type>& dv)
        : data_(thrust::raw_pointer_cast(dv.data())), n(dv.size())
    {
    }
    __host__ ArrayView(thrust::device_vector<typename std::remove_const<T>::type> const& dv)
        : data_(const_cast<T*>(thrust::raw_pointer_cast(dv.data()))), n(dv.size())
    {
    }
#endif


    template <size_t N>
    HD ArrayView(T (&arr)[N]) : data_(arr), n(N)
    {
    }

    template <typename Cont, typename = typename std::enable_if<
                                 std::is_convertible<decltype(std::declval<Cont>().data()), T*>::value &&
                                 !std::is_same<Cont, ArrayView<T>>::value>::type>
    ArrayView(Cont& dv) : data_(dv.data()), n(dv.size())
    {
    }

    HD reference operator[](size_t id) const SAIGA_NOEXCEPT { return data_[id]; }

    HD reference back() SAIGA_NOEXCEPT { return data_[n - 1]; }

    HD const_reference back() const SAIGA_NOEXCEPT { return data_[n - 1]; }


    HD pointer data() const SAIGA_NOEXCEPT { return data_; }

    HD size_type size() const SAIGA_NOEXCEPT { return n; }
    HD size_type byte_size() const SAIGA_NOEXCEPT { return sizeof(T) * n; }

    HD iterator begin() const SAIGA_NOEXCEPT { return data_; }

    HD iterator end() const SAIGA_NOEXCEPT { return data_ + n; }


#ifdef SAIGA_GENERATE_THRUST_CONSTRUCTOR
    thrust::device_ptr<T> tbegin() const { return thrust::device_pointer_cast(begin()); }
    thrust::device_ptr<T> tend() const { return thrust::device_pointer_cast(end()); }
#endif

    // remove elements from the right and left
    HD ArrayView<T> slice(size_t left, size_t right) const { return ArrayView<T>(data_ + left, n - right - left); }

    HD ArrayView<T> slice_n(size_t offset, size_t n) const { return ArrayView<T>(data_ + offset, n); }

    //    HD operator T*() const SAIGA_NOEXCEPT{
    //        return data_;
    //    }

   private:
    T* data_;
    size_type n;
};

template <typename T>
HD ArrayView<T> make_ArrayView(T* data_, size_t n)
{
    return ArrayView<T>(data_, n);
}


template <typename T, size_t N>
HD ArrayView<T> make_ArrayView(T (&arr)[N])
{
    return ArrayView<T>(arr);
}

template <typename Container>
auto make_ArrayView(Container const& cont) -> ArrayView<typename std::remove_reference<decltype(*cont.data())>::type>
{
    using type = typename std::remove_reference<decltype(*cont.data())>::type;
    return ArrayView<type>(cont);
}

}  // namespace Saiga
