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
    class iterator : public std::iterator<std::input_iterator_tag,  // iterator_category
                                          IndexType                 // reference
                                          >
    {
        IndexType num;

       public:
        explicit iterator(IndexType _num = 0) : num(_num) {}
        iterator& operator++()
        {
            num = num + 1;
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

}  // namespace Saiga
