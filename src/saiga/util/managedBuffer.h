/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/assert.h"
#include <vector>

namespace Saiga {

template<typename T>
class SAIGA_TEMPLATE ManagedBuffer
{
protected:
    std::vector<char> buffer;
    std::vector<char> constructed;
    size_t capacity = 0;


    T* getPtr(int id){
        char* ptr = buffer.data() + id * sizeof(T);
        return reinterpret_cast<T*>(ptr);
    }

    T& get(int id){
        return *getPtr(id);
    }

public:


    ManagedBuffer(size_t capacity) :
        buffer(capacity*sizeof(T)),
        constructed(capacity,0),
        capacity(capacity) {
    }


    void resize(size_t size)
    {
        //TODO
        SAIGA_ASSERT(0);
    }


    bool isConstructed(int id)
    {
        return constructed[id];
    }


    T& operator[](int id)
    {
        if(!isConstructed(id))
            create(id);
        return get(id);
    }


    void free(int id)
    {
        SAIGA_ASSERT(isConstructed(id));
        get(id).~T();
        constructed[id] = false;
    }

    void create(int id)
    {
        if(isConstructed(id))
        {
            free(id);
        }
        auto ptr = getPtr(id);
        new (ptr) T ();
        constructed[id] = true;
    }

    void freeAll()
    {
        for(int i = 0;i < capacity; ++i)
        {
            if(isConstructed(i))
                free(i);
        }
    }

};

}
