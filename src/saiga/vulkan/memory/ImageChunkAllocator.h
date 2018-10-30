//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once
#include "BaseChunkAllocator.h"

namespace Saiga{
namespace Vulkan{
namespace Memory{

class ImageChunkAllocator : public BaseChunkAllocator {
protected:
    ChunkIterator createNewChunk() override;
};

}
}
}

