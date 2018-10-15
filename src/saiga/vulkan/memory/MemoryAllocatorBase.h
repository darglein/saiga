//
// Created by Peter Eichinger on 15.10.18.
//

#pragma once
#include "MemoryLocation.h"
namespace Saiga{
namespace Vulkan{
namespace Memory{

struct SAIGA_GLOBAL MemoryAllocatorBase {
    virtual MemoryLocation allocate(vk::DeviceSize size) = 0;
};

}
}
}