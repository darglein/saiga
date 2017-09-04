/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/common.h"

#include <thrust/device_vector.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Saiga {

//Thanks to Johannes Pieger

template<typename T>
struct array_view{
    using value_type = T;
    using reference = value_type&;
    using const_reference = value_type const&;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using iterator = pointer;
    using const_iterator = const_pointer;
//    using size_type = size_t;
    using size_type = uint32_t; //32 bit integer are faster in cuda!
    using difference_type = ptrdiff_t;

    HD array_view() : data_(nullptr), n(0){}
    HD array_view(T* data_, size_t n) : data_(data_), n(n){}

    HD array_view(array_view<T> const&) = default;
    HD array_view& operator=(array_view<T> const&) = default;

    __host__ array_view(thrust::device_vector<typename std::remove_const<T>::type>& dv)
        : data_(thrust::raw_pointer_cast(dv.data())),
          n(dv.size())
    {}
    __host__ array_view(thrust::device_vector<typename std::remove_const<T>::type> const& dv)
        : data_(const_cast<T*>(thrust::raw_pointer_cast(dv.data()))),
          n(dv.size())
    {}


    template<size_t N>
    HD array_view(T(&arr)[N]) : data_(arr), n(N){}

    template<typename Cont,
             typename = typename std::enable_if<
                 std::is_convertible<decltype(std::declval<Cont>().data()), T*>::value
                 && !std::is_same<Cont, array_view<T>>::value
                 >::type>
    array_view(Cont& dv) : data_(dv.data()), n(dv.size()){}

	HD reference operator[](size_t id) const SAIGA_NOEXCEPT{
        //			if(id >= n) {
        //				printf("invalid access: id %lu >= size %lu\n", id, n);
        //				assert(false);
        //			}
        return data_[id];
    }

		HD reference back() SAIGA_NOEXCEPT{
        return data_[n-1];
    }

		HD const_reference back() const SAIGA_NOEXCEPT{
        return data_[n-1];
    }


		HD pointer data() const SAIGA_NOEXCEPT{
        return data_;
    }

		HD size_type size() const SAIGA_NOEXCEPT{
        return n;
    }
		HD size_type byte_size() const SAIGA_NOEXCEPT{
        return sizeof(T) * n;
    }

		HD iterator begin() const SAIGA_NOEXCEPT{
        return data_;
    }

		HD iterator end() const SAIGA_NOEXCEPT{
        return data_+n;
    }

    thrust::device_ptr<T> tbegin() const {
        return thrust::device_pointer_cast(begin());
    }
    thrust::device_ptr<T> tend() const {
        return thrust::device_pointer_cast(end());
    }

    //remove elements from the right and left
    HD array_view<T> slice(size_t left, size_t right) const {
        return array_view<T>(data_ + left, n - right - left);
    }

    HD array_view<T> slice_n(size_t offset, size_t n) const {
        return array_view<T>(data_ + offset, n);
    }


//    HD operator bool() const SAIGA_NOEXCEPT {
//        return data_;
//    }

	HD operator T*() const SAIGA_NOEXCEPT{
        return data_;
    }


private:
    T* data_;
    size_type n;
};

template<typename T>
HD array_view<T> make_array_view(T* data_, size_t n) {
    return array_view<T>(data_, n);
}


template<typename T, size_t N>
HD array_view<T> make_array_view(T(&arr)[N]) {
    return array_view<T>(arr);
}

template<typename Container>
auto make_array_view(Container const& cont) -> array_view<typename std::remove_reference<decltype(*cont.data())>::type> {
    using type = typename std::remove_reference<decltype(*cont.data())>::type;
    return array_view<type>(cont);
}

}
