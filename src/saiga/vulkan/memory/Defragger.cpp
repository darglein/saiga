//
// Created by Peter Eichinger on 2019-01-21.
//

#include "Defragger.h"

#include <saiga/util/threadName.h>

namespace Saiga::Vulkan::Memory
{
static std::atomic_int counter = 0;

void Defragger::start()
{
    if (!enabled || running)
    {
        return;
    }
    defrag_operations.clear();
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        running = true;
    }
    start_condition.notify_one();
}

void Defragger::stop()
{
    if (!running)
    {
        return;
    }
    running = false;
    {
        std::unique_lock<std::mutex> lock(running_mutex);
    }
}


void Defragger::worker_func()
{
    Saiga::setThreadName("Defragger " + std::to_string(counter++));
    while (true)
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_condition.wait(lock, [this] { return running || quit; });
        if (quit)
        {
            return;
        }
        std::unique_lock<std::mutex> running_lock(running_mutex);
        if (allocator->chunks.empty())
        {
            continue;
        }

        apply_invalidations();

        run();

        running = false;
    }
}

void Defragger::run()
{
    find_defrag_ops();
    perform_defrag();
}

void Defragger::find_defrag_ops()
{
    auto& chunks = allocator->chunks;

    for (auto chunk_iter = chunks.rbegin(); running && chunk_iter != chunks.rend(); ++chunk_iter)
    {
        auto& allocs = chunk_iter->allocations;
        for (auto alloc_iter = allocs.rbegin(); running && alloc_iter != allocs.rend(); ++alloc_iter)
        {
            auto& source = **alloc_iter;

            auto begin = chunks.begin();
            auto end   = (chunk_iter).base();  // Conversion from reverse to normal iterator moves one back
            //
            auto new_place = allocator->strategy->findRange(begin, end, source.size);

            if (new_place.first != end)
            {
                const auto target_iter = new_place.second;
                const auto& target     = *target_iter;

                auto current_chunk = (chunk_iter + 1).base();


                if (current_chunk != new_place.first || target.offset < (**alloc_iter).offset)
                {
                    auto weight =
                        get_operation_penalty(new_place.first, target_iter, current_chunk, (alloc_iter + 1).base());

                    defrag_operations.insert(DefragOperation{&source, new_place.first->chunk->memory, target, weight});
                }
            }
        }
    }
}

void Defragger::perform_defrag()
{
    {
        using namespace std::chrono_literals;
    }
    for (auto op = defrag_operations.begin(); running && op != defrag_operations.end(); ++op)
    {
        if (allocator->memory_is_free(op->targetMemory, op->target))
        {
            LOG(INFO) << "DEFRAG" << *(op->source) << "->" << op->targetMemory << "," << op->target.offset << " "
                      << op->target.size;

            MemoryLocation* reserve_space = allocator->reserve_space(op->targetMemory, op->target, op->source->size);
            auto defrag_cmd               = allocator->queue->commandPool.createAndBeginOneTimeBuffer();

            op->source->copy_to(defrag_cmd, reserve_space);

            defrag_cmd.end();

            allocator->queue->submitAndWait(defrag_cmd);

            allocator->queue->commandPool.freeCommandBuffer(defrag_cmd);

            allocator->move_allocation(reserve_space, op->source);
        }
    }
}



float Defragger::get_operation_penalty(ConstChunkIterator target_chunk, ConstFreeIterator target_location,
                                       ConstChunkIterator source_chunk, ConstAllocationIterator source_location) const
{
    auto& source_ptr = *source_location;
    float weight     = 0;


    if (target_chunk == source_chunk)
    {
        // move inside a chunk should be done after moving to others
        weight += penalties.same_chunk;
    }

    // if the move creates a hole that is smaller than the memory chunk itself -> add weight
    if (target_location->size != source_ptr->size && (target_location->size - source_ptr->size < source_ptr->size))
    {
        weight += penalties.target_small_hole * (1 - (static_cast<float>(source_ptr->size) / target_location->size));
    }

    // If move creates a hole at source -> add weight
    auto next = std::next(source_location);
    if (source_location != source_chunk->allocations.cbegin() && next != source_chunk->allocations.cend())
    {
        auto& next_ptr = *next;
        auto& prev     = *std::prev(source_location);

        if (source_ptr->offset == prev->offset + prev->size &&
            source_ptr->offset + source_ptr->size == next_ptr->offset)
        {
            weight += penalties.source_create_hole;
        }
    }

    // Penalty if allocation is not the last allocation in chunk
    if (next != source_chunk->allocations.cend())
    {
        weight += penalties.source_not_last_alloc;
    }

    if (std::next(source_chunk) != allocator->chunks.cend())
    {
        weight += penalties.source_not_last_chunk;
    }

    return weight;
}

void Defragger::apply_invalidations()
{
    std::unique_lock<std::mutex> invalidate_lock(invalidate_mutex);
    if (!defrag_operations.empty() && !invalidate_set.empty())
    {
        auto ops_iter = defrag_operations.begin();

        while (ops_iter != defrag_operations.end())
        {
            auto target_mem = ops_iter->targetMemory;
            if (invalidate_set.find(target_mem) != invalidate_set.end())
            {
                ops_iter = defrag_operations.erase(ops_iter);
            }
            else
            {
                ++ops_iter;
            }
        }
        invalidate_set.clear();
    }
}

void Defragger::invalidate(vk::DeviceMemory memory)
{
    std::unique_lock<std::mutex> invalidate_lock(invalidate_mutex);
    invalidate_set.insert(memory);
}

void Defragger::invalidate(MemoryLocation* location)
{
    std::unique_lock<std::mutex> invalidate_lock(invalidate_mutex);
    for (auto op_iter = defrag_operations.begin(); op_iter != defrag_operations.end();)
    {
        if (op_iter->source == location)
        {
            op_iter = defrag_operations.erase(op_iter);
        }
        else
        {
            ++op_iter;
        }
    }
}
}  // namespace Saiga::Vulkan::Memory