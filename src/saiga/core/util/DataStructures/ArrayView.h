/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

#include <cstddef>
#include <cstdint>

#include <type_traits>

#ifdef SAIGA_ARRAY_VIEW_THRUST
#    include "saiga/cuda/thrust_helper.h"
#endif

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

    HD ArrayView() : data_(nullptr), n(0) {}
    HD ArrayView(T* data_, size_t n) : data_(data_), n(n) {}

    // Initialize from a single value
    HD ArrayView(T& element) : data_(&element), n(1) {}

    ArrayView(ArrayView<T> const&) = default;
    ArrayView& operator=(ArrayView<T> const&) = default;

    // Allow assignment of different arrayviews as long as the pointers are compatible
    template <typename G>
    ArrayView(ArrayView<G> const& av) : data_(av.data()), n(av.size())
    {
    }

    template <size_t N>
    HD ArrayView(T (&arr)[N]) : data_(arr), n(N)
    {
    }


    template <typename Cont,
              typename std::enable_if<std::is_convertible<decltype(std::declval<Cont>().data()), T*>::value &&
                                          std::is_convertible<decltype(std::declval<Cont>().size()), size_t>::value &&
                                          !std::is_same<Cont, ArrayView<T>>::value,
                                      int>::type = 0>
    ArrayView(Cont& dv) : data_(dv.data()), n(dv.size())
    {
    }


#ifdef SAIGA_ARRAY_VIEW_THRUST
    ArrayView(thrust::device_vector<T>& vector) : data_(vector.data().get()), n(vector.size()) {}
    ArrayView(const thrust::device_vector<T>& vector) : data_(vector.data().get()), n(vector.size()) {}
#else
    // For thrust device vectors which use .data().get() to get the raw pointer.
    template <typename Cont,
              typename std::enable_if<std::is_convertible<decltype(std::declval<Cont>().data().get()), T*>::value &&
                                          std::is_convertible<decltype(std::declval<Cont>().size()), size_t>::value &&
                                          !std::is_same<Cont, ArrayView<T>>::value,
                                      int>::type = 0>
    ArrayView(Cont& dv) : data_(dv.data().get()), n(dv.size())
    {
    }
#endif

    HD reference operator[](size_t id) const SAIGA_NOEXCEPT { return data_[id]; }


    HD reference front() const SAIGA_NOEXCEPT { return data_[0]; }
    HD reference back() SAIGA_NOEXCEPT { return data_[n - 1]; }
    HD const_reference back() const SAIGA_NOEXCEPT { return data_[n - 1]; }


    HD pointer data() const SAIGA_NOEXCEPT { return data_; }

    HD size_type size() const SAIGA_NOEXCEPT { return n; }
    HD size_type byte_size() const SAIGA_NOEXCEPT { return sizeof(T) * n; }

    HD iterator begin() const SAIGA_NOEXCEPT { return data_; }

    HD iterator end() const SAIGA_NOEXCEPT { return data_ + n; }



    // remove elements from the right and left
    HD ArrayView<T> slice(size_t left, size_t right) const { return ArrayView<T>(data_ + left, n - right - left); }

    HD ArrayView<T> slice_n(size_t offset, size_t n) const { return ArrayView<T>(data_ + offset, n); }

    // returns the first n elemetns
    HD ArrayView<T> head(size_t n) const { return ArrayView<T>(data_, n); }

    // returns the last n elements
    HD ArrayView<T> tail(size_t n2) const { return ArrayView<T>(data_ + n - n2, n2); }

    HD bool empty() const { return n == 0; }

    HD void pop_back() { --n; }

    HD bool isAligned() { return (((uintptr_t)data_) % (alignof(T))) == 0; }
    HD bool isAligned(size_t alignment) { return (((uintptr_t)data_) % (alignment)) == 0; }

#ifdef SAIGA_ARRAY_VIEW_THRUST
    thrust::device_ptr<T> device_begin() const { return thrust::device_ptr<T>(begin()); }
    thrust::device_ptr<T> device_end() const { return thrust::device_ptr<T>(end()); }
#endif

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
