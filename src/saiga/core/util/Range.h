/**
 * Copyright (c) 2017 Darius Rückert
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
template <typename IndexType = int>
class Range
{
   public:
    // member typedefs provided through inheriting from std::iterator
    class iterator
    {
        IndexType num;

       public:
        explicit iterator(IndexType _num) : num(_num) {}
        iterator& operator++()
        {
            ++num;
            return *this;
        }
        iterator operator++(int)
        {
            iterator retval = *this;
            ++(*this);
            return retval;
        }
        bool operator==(iterator other) const { return num == other.num; }
        bool operator!=(iterator other) const { return !(*this == other); }
        IndexType operator*() const { return num; }
    };

    Range(IndexType _from, IndexType _to) : from(_from), to(_to) {}
    iterator begin() { return iterator(from); }
    iterator end() { return iterator(to); }

    IndexType from, to;
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
