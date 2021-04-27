/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

namespace Saiga
{
template <typename C>
class SAIGA_TEMPLATE Singleton
{
   public:
    static C* instance()
    {
        static C _instance;
        return &_instance;
    }
    virtual ~Singleton() {}

   private:
   protected:
    Singleton() {}
};

}  // namespace Saiga
