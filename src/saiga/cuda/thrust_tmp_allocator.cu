/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "thrust_tmp_allocator.h"

namespace Saiga
{
memory_resource::memory_resource() : thrust::mr::memory_resource<>() {}

void* memory_resource::do_allocate(std::size_t num_bytes, std::size_t alignment)
{
    SAIGA_ASSERT(!locked);
    locked = true;

    if (mem.size() < num_bytes + alignment)
    {
        mem.resize(num_bytes + alignment);
    }
    char* ptr = (char*)mem.data().get();
    ptr += (alignment - (unsigned long)ptr % alignment) % alignment;
    return ptr;
}

void memory_resource::do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment)
{
    locked = false;
}

}  // namespace Saiga
