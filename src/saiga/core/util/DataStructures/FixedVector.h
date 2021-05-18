/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/util/Align.h"
#include "saiga/core/util/assert.h"

namespace Saiga
{
/**
 * A Vector of elements of fixed capacity.
 * Usefull for objects which are not copyable/moveable.
 */
template <typename T, size_t Alignment = 64>
class FixedVectorHeap
{
   public:
    FixedVectorHeap(size_t capacity) : _capacity(capacity)
    {
        _data = reinterpret_cast<T*>(allocator.allocate(sizeof(T) * capacity));
    }
    ~FixedVectorHeap()
    {
        for (auto i = 0; i < _size; ++i) _data[i].~T();
        allocator.deallocate(reinterpret_cast<char*>(_data), sizeof(T) * _capacity);
    }

    template <typename... Args>
    size_t emplace_back(Args&&... args)
    {
        SAIGA_ASSERT(_size < capacity());
        auto i = _size++;
        new (&_data[i]) T(std::forward<Args>(args)...);
        return i;
    }

    void pop_back()
    {
        SAIGA_ASSERT(_size > 0);
        auto i = --_size;
        _data[i].~T();
    }

    T& back() { return data()[_size - 1]; }

    constexpr auto capacity() { return _capacity; }

    constexpr T& operator[](size_t i)
    {
        SAIGA_DEBUG_ASSERT(i < _size);
        return _data[i];
    }

    constexpr T* begin() { return _data; }
    constexpr T* end() { return _data + _size; }

    // returns the amount of memory required
    constexpr size_t memory() { return sizeof(this) + sizeof(T) * _capacity; }

    constexpr T* data() { return _data; }
    constexpr auto size() { return _size; }

   private:
    size_t _size = 0;
    size_t _capacity;
    T* _data;

    aligned_allocator<char, Alignment> allocator;
};

template <typename T, size_t _capacity>
class FixedVectorStack
{
   public:
    FixedVectorStack() {}
    ~FixedVectorStack() { clear(); }

    template <typename... Args>
    size_t emplace_back(Args&&... args)
    {
        SAIGA_ASSERT(_size < capacity());
        auto i = _size++;
        new (&data()[i]) T(std::forward<Args>(args)...);
        return i;
    }

    void pop_back()
    {
        SAIGA_ASSERT(_size > 0);
        auto i = --_size;
        data()[i].~T();
    }

    constexpr auto capacity() { return _capacity; }

    constexpr T& operator[](size_t i)
    {
        SAIGA_DEBUG_ASSERT(i < _size);
        return data()[i];
    }

    void clear()
    {
        for (auto i = 0; i < _size; ++i) data()[i].~T();
        _size = 0;
    }
    constexpr bool empty() { return _size == 0; }
    constexpr T& back() { return data()[_size - 1]; }

    constexpr T* begin() { return data(); }
    constexpr T* end() { return data() + _size; }

    // returns the amount of memory required
    constexpr size_t memory() { return sizeof(this) + sizeof(T) * _capacity; }

    constexpr T* data() { return reinterpret_cast<T*>(_data); }
    constexpr auto size() { return _size; }

   private:
    alignas(alignof(T)) char _data[_capacity * sizeof(T)];
    size_t _size = 0;
};


}  // namespace Saiga
