/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/fileChecker.h"

#include <tuple>
#include <vector>

namespace Saiga
{
struct SAIGA_CORE_API NoParams
{
};

SAIGA_CORE_API inline bool operator==(const NoParams&, const NoParams&)
{
    return true;
}


template <typename KeyType, typename DataType, typename ParamType = NoParams>
class ObjectCache
{
   public:
    using InternalType = std::tuple<KeyType, ParamType, DataType>;
    std::vector<InternalType> objects;

   public:
    virtual ~ObjectCache() { clear(); }
    void clear() { objects.clear(); }


    void put(const KeyType& key, const DataType& obj, const ParamType& params = ParamType())
    {
        SAIGA_ASSERT(!exists(key, params));
        objects.emplace_back(key, params, obj);
    }

    bool get(const KeyType& key, DataType& obj, const ParamType& params = ParamType())
    {
        for (auto& data : objects)
        {
            if (std::get<0>(data) == key && std::get<1>(data) == params)
            {
                obj = std::get<2>(data);
                return true;
            }
        }
        return false;
    }

    bool exists(const KeyType& key, const ParamType& params = ParamType())
    {
        for (auto& data : objects)
        {
            if (std::get<0>(data) == key && std::get<1>(data) == params)
            {
                return true;
            }
        }
        return false;
    }
};


}  // namespace Saiga
