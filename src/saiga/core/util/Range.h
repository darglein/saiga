/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <algorithm>

namespace Saiga
{
// Initial source:
// https://en.cppreference.com/w/cpp/iterator/iterator
// Modifications:
// - template IndexType
// - from/to member variables
template <typename IndexType = int, typename Comp = std::not_equal_to<IndexType>>
class Range
{
   public:
    // member typedefs provided through inheriting from std::iterator
    class iterator
    {
        IndexType num;
        IndexType stride;

       public:
        HD inline explicit iterator(IndexType _num, IndexType _stride) : num(_num), stride(_stride) {}

        // pre-increment
        HD inline iterator& operator++()
        {
            num += stride;
            return *this;
        }

        HD inline bool operator!=(iterator other) const
        {
            Comp ne;
            return ne(num, other.num);
        }
        HD inline IndexType operator*() const { return num; }
    };

    HD inline Range(IndexType _from, IndexType _to, IndexType _stride = 1) : from(_from), to(_to), stride(_stride) {}
    HD inline iterator begin() { return iterator(from, stride); }
    HD inline iterator end() { return iterator(to, stride); }
    HD inline Range<IndexType, Comp> inverse()
    {
        static_assert(std::is_same<Comp, std::not_equal_to<IndexType>>::value,
                      "Inverse only valid for not_equal comparison!");
        return Range<IndexType, Comp>(to - stride, from - stride, -stride);
    }

    IndexType from, to, stride;
};


template <typename IndexType = int>
class StridedRange
{
   public:
    class iterator
    {
        IndexType num;
        IndexType stride;

       public:
        HD inline explicit iterator(IndexType _num, IndexType _stride) : num(_num), stride(_stride) {}

        // pre-increment
        HD inline iterator& operator++()
        {
            num += stride;
            return *this;
        }

        HD inline bool operator!=(iterator other) const { return num < other.num; }
        HD inline IndexType operator*() const { return num; }
    };

    HD inline StridedRange(IndexType _from, IndexType _to, IndexType _stride = 1)
        : from(_from), to(_to), stride(_stride)
    {
    }
    HD inline iterator begin() { return iterator(from, stride); }
    HD inline iterator end() { return iterator(to, stride); }

    IndexType from, to, stride;
};


/**
 * An indirect range iterator for index-value iteration.
 * Example:
 *
 * //Contains only the indices of the valid elements of 'data'
 * std::vector<int> validIndices;
 * std::vector<T> data;
 *
 * auto ir = IndirectRange(validIndices.begin(),validIndices.end(),data.data());
 * for(auto&& t : ir)
 *      t.doSomething();
 */
template <typename T, typename KeyIterator>
class IndirectRange
{
   public:
    class iterator : public std::iterator<std::input_iterator_tag, T>
    {
        KeyIterator it;
        T* value;

       public:
        explicit iterator(KeyIterator _it, T* _value) : it(_it), value(_value) {}
        iterator& operator++()
        {
            ++it;
            return *this;
        }
        iterator operator++(int)
        {
            iterator retval = *this;
            ++(*this);
            return retval;
        }
        bool operator==(iterator other) const { return it == other.it; }
        bool operator!=(iterator other) const { return !(*this == other); }
        auto& operator*() const { return value[*it]; }
    };

    IndirectRange(KeyIterator _from, KeyIterator _to, T* _value) : from(_from), to(_to), value(_value) {}
    iterator begin() { return iterator(from, value); }
    iterator end() { return iterator(to, value); }

    KeyIterator from, to;
    T* value;
};

}  // namespace Saiga
