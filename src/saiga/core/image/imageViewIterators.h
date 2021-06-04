/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/imath.h"
#include "saiga/core/util/assert.h"


namespace Saiga
{
/**
 * Iterates over an image row.
 * Dereferencing it returns the actual image pixel.
 */
template <typename T>
struct RowIterator
{
    RowIterator(T* ptr, int row, int col) : _ptr(ptr), row(row), col(col) {}


    int x() { return col; }
    int y() { return row; }
    T& value() { return *_ptr; }

    RowIterator<T> operator*() { return *this; }

    RowIterator<T> operator++()
    {
        ++_ptr;
        ++col;
        return *this;
    }
    bool operator!=(const RowIterator<T> other) const { return col != other.col; }

   private:
    T* _ptr;
    int row;
    int col;
};


/**
 * One row of the image.
 */
template <typename T>
struct ImageRow
{
    ImageRow(int row, T* begin, int width) : _row(row), _begin(begin), width(width) {}

    size_t size() { return width; }

    T& operator()(int i) { return _begin[i]; }
    T& operator[](int i) { return _begin[i]; }
    T& operator()(int i) const { return _begin[i]; }
    T& operator[](int i) const { return _begin[i]; }

    RowIterator<T> begin() { return RowIterator<T>(_begin, _row, 0); }
    RowIterator<T> end() { return RowIterator<T>(_begin + width, _row, width); }

   private:
    int _row;
    T* _begin;
    int width;
};



/**
 * Iterates over an image view row by row.
 * Dereferenceing it returns a 'ImageRow' object which again can be iterated over.
 */
template <typename ViewType>
struct ImageIteratorRowmajor
{
    using ValueType = typename ViewType::Type;

    ImageIteratorRowmajor(ViewType& img, int row) : img(img), row(row) {}



    ImageRow<ValueType> operator*() { return ImageRow<ValueType>(row, img.rowPtr(row), img.w); }
    ImageIteratorRowmajor<ViewType> operator++()
    {
        ++row;
        return *this;
    }
    bool operator!=(const ImageIteratorRowmajor<ViewType> other) const { return row != other.row; }

   private:
    ViewType& img;
    int row;
};


}  // namespace Saiga
