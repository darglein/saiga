/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/util/Align.h"
#include "saiga/core/util/assert.h"

#include <new>
#include <utility>

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
    ~FixedVectorHeap() { release(); }

    // Deep copy, requires a copyable T. The copy has the same capacity.
    FixedVectorHeap(const FixedVectorHeap& other) : FixedVectorHeap(other._capacity)
    {
        for (size_t i = 0; i < other._size; ++i) emplace_back(other._data[i]);
    }

    // Takes over the memory of 'other', which is empty with capacity 0 afterwards. The elements are not moved.
    FixedVectorHeap(FixedVectorHeap&& other) noexcept
        : _size(other._size), _capacity(other._capacity), _data(other._data)
    {
        other._size     = 0;
        other._capacity = 0;
        other._data     = nullptr;
    }

    FixedVectorHeap& operator=(const FixedVectorHeap& other)
    {
        if (this != &other)
        {
            FixedVectorHeap tmp(other);
            swap(tmp);
        }
        return *this;
    }

    FixedVectorHeap& operator=(FixedVectorHeap&& other) noexcept
    {
        if (this != &other)
        {
            release();
            _size           = other._size;
            _capacity       = other._capacity;
            _data           = other._data;
            other._size     = 0;
            other._capacity = 0;
            other._data     = nullptr;
        }
        return *this;
    }

    void swap(FixedVectorHeap& other) noexcept
    {
        std::swap(_size, other._size);
        std::swap(_capacity, other._capacity);
        std::swap(_data, other._data);
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
    constexpr size_t memory() { return sizeof(*this) + sizeof(T) * _capacity; }

    constexpr T* data() { return _data; }
    constexpr auto size() { return _size; }

   private:
    size_t _size = 0;
    size_t _capacity;
    T* _data;

    void release()
    {
        if (!_data) return;
        for (size_t i = 0; i < _size; ++i) _data[i].~T();
        allocator.deallocate(reinterpret_cast<char*>(_data), sizeof(T) * _capacity);
        _data = nullptr;
        _size = 0;
    }

    aligned_allocator<char, Alignment> allocator;
};

template <typename T, size_t _capacity>
class FixedVectorStack
{
   public:
    FixedVectorStack() {}
    ~FixedVectorStack() { clear(); }

    // The elements are copy constructed. The implicit copy would copy the raw bytes and destroy them twice.
    FixedVectorStack(const FixedVectorStack& other)
    {
        for (size_t i = 0; i < other._size; ++i) emplace_back(other.data()[i]);
    }

    FixedVectorStack& operator=(const FixedVectorStack& other)
    {
        if (this != &other)
        {
            clear();
            for (size_t i = 0; i < other._size; ++i) emplace_back(other.data()[i]);
        }
        return *this;
    }

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
        for (size_t i = 0; i < _size; ++i) data()[i].~T();
        _size = 0;
    }
    constexpr bool empty() { return _size == 0; }
    constexpr T& back() { return data()[_size - 1]; }

    constexpr T* begin() { return data(); }
    constexpr T* end() { return data() + _size; }

    // returns the amount of memory required
    constexpr size_t memory() { return sizeof(*this); }

    constexpr T* data() { return reinterpret_cast<T*>(_data); }
    constexpr const T* data() const { return reinterpret_cast<const T*>(_data); }
    constexpr auto size() { return _size; }

   private:
    alignas(alignof(T)) char _data[_capacity * sizeof(T)];
    size_t _size = 0;
};


}  // namespace Saiga
