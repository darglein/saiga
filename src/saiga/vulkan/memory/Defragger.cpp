//
// Created by Peter Eichinger on 2019-01-21.
//

#include "Defragger.h"

#include "saiga/core/util/threadName.h"
#include "saiga/vulkan/Base.h"

#include "ImageCopyComputeShader.h"

#include <optional>
namespace Saiga::Vulkan::Memory
{
template <typename T>
bool Defragger<T>::perform_free_operations()
{
    int current_frame = base->current_frame;

    for (auto current = free_operations.begin(); running && current != free_operations.end();)
    {
        auto delay = current->delay;
        if (delay < current_frame)
        {
            if (current->target)
            {
                allocator->deallocate(current->target);
                current->source->mark_dynamic();
            }
            current = free_operations.erase(current);
        }
        else
        {
            current++;
        }
    }
    auto should_run = !free_operations.empty();
    return should_run;
}


template <typename T>
void Defragger<T>::start()
{
    if (!valid || !enabled || running)
    {
        return;
    }
    // defrag_operations.clear();
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        running = true;
    }
    start_condition.notify_one();
}

template <typename T>
void Defragger<T>::stop()
{
    if (!valid || !running)
    {
        return;
    }
    running = false;
    {
        std::unique_lock<std::mutex> lock(running_mutex);
    }
}


template <typename T>
void Defragger<T>::worker_func()
{
    Saiga::setThreadName("Defragger");
    if (!valid)
    {
        return;
    }
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

template <typename T>
void Defragger<T>::run()
{
    bool performed = true;
    while (running && performed)
    {
        performed = false;
        find_defrag_ops();
        performed |= perform_defrag();
        performed |= perform_free_operations();
    }
}



template <typename T>
void Defragger<T>::find_defrag_ops()
{
    // TODO: REFACTOR THIS!!!
    auto& chunks = allocator->chunks;

    // contains a vector for each chunk, which in turn contains the free memory between the allocations. contains 0 if
    // they are back to back.
    std::vector<std::vector<vk::DeviceSize>> all_free;

    all_free.resize(chunks.size());
    for (int i = 0; running && i < chunks.size(); ++i)
    {
        auto& freeSizes = all_free[i];
        auto& allocs    = chunks[i].allocations;
        freeSizes.resize(allocs.size() + 1);

        if (!allocs.empty())
        {
            auto freeIter = freeSizes.begin();
            auto curr     = allocs.begin();

            while (curr != allocs.end())
            {
                vk::DeviceSize free_before;

                if (curr == allocs.begin())
                {
                    free_before = begin(curr);
                }
                else
                {
                    free_before = begin(curr) - end(std::prev(curr));
                }
                *freeIter = free_before;

                freeIter++;
                curr++;
            }

            freeSizes[allocs.size()] = chunks[i].size - end(allocs.end() - 1);
        }
        else
        {
            freeSizes[0] = chunks[i].size;
        }
    }

    for (auto chunk_iter = chunks.rbegin(); running && chunk_iter != chunks.rend(); ++chunk_iter)
    {
        const auto source_chunk_index = static_cast<uint32_t>(std::distance(chunk_iter, chunks.rend()) - 1);

        const auto source_chunk_size = chunk_iter->size;
        auto& allocs                 = chunk_iter->allocations;

        if (allocs.empty())
        {
            continue;
        }


        auto& freeSizes = all_free[source_chunk_index];

        // auto curr     = allocs.cbegin();
        auto freeIter = freeSizes.cbegin();
        for (auto curr = allocs.cbegin(); curr != allocs.cend() && running; ++curr, ++freeIter)
        {
            if (get(curr)->is_static())
            {
                continue;
            }
            auto found = std::find_if(free_operations.begin(), free_operations.end(),
                                      [=](const FreeOperation& op) { return op.source == get(curr); });
            if (found != free_operations.end())
            {
                continue;
            }
            auto alloc_index               = std::distance(allocs.cbegin(), curr);
            vk::DeviceSize size_if_removed = *freeIter + *std::next(freeIter) + size(curr);

            // Get the weight of the current allocation and its neighbours
            float weight_prev        = 0.0f;
            float weight_remove_prev = 0.0f;

            if (curr != allocs.begin())
            {
                weight_prev =
                    get_weight(source_chunk_index, source_chunk_size, std::prev(curr), *std::prev(freeIter), *freeIter);
                weight_remove_prev = get_weight(source_chunk_index, source_chunk_size, std::prev(curr),
                                                *std::prev(freeIter), size_if_removed);
            }

            float weight_next        = 0.0f;
            float weight_remove_next = 0.0f;

            if (std::next(curr) != allocs.end())
            {
                weight_next = get_weight(source_chunk_index, source_chunk_size, std::next(curr), *std::next(freeIter),
                                         *std::next(freeIter, 2));
                weight_remove_next = get_weight(source_chunk_index, source_chunk_size, std::next(curr), size_if_removed,
                                                *std::next(freeIter, 2));
            }

            float weight_curr =
                get_weight(source_chunk_index, source_chunk_size, curr, *freeIter, *std::next(freeIter));

            float weight_before = weight_prev + weight_curr + weight_next;
            float weight_after  = weight_remove_prev + weight_remove_next;

            float min_weight = std::numeric_limits<float>::infinity();

            auto targets_found = 0;
            std::pair<int, FreeListEntry> target;
            for (int t_chunk = 0; targets_found < config.max_targets && t_chunk <= source_chunk_index; ++t_chunk)
            {
                auto& chunk      = chunks[t_chunk];
                auto& free_chunk = all_free[t_chunk];
                auto& t_allocs   = chunks[t_chunk].allocations;
                auto alloc_iter  = t_allocs.cbegin();

                for (auto free_index = 0U; targets_found < config.max_targets && free_index < free_chunk.size();
                     ++free_index)
                {
                    if (t_chunk == source_chunk_index && free_index > alloc_index)
                    {
                        break;
                    }
                    auto free_size = free_chunk[free_index];

                    if (free_size >= size(curr))
                    {
                        float t_weight_prev          = 0.0f;
                        float t_weight_prev_inserted = 0.0f;
                        float t_weight_next          = 0.0f;
                        float t_weight_next_inserted = 0.0f;
                        if (free_index != 0)
                        {
                            t_weight_prev = get_weight(t_chunk, source_chunk_size, std::prev(alloc_iter),
                                                       free_chunk[free_index - 1], free_chunk[free_index]);
                            // the inserted allocation will align with the previous allocation
                            t_weight_prev_inserted = get_weight(t_chunk, source_chunk_size, std::prev(alloc_iter),
                                                                free_chunk[free_index - 1], 0);
                        }
                        if (free_index != free_chunk.size() - 1)
                        {
                            t_weight_next = get_weight(t_chunk, source_chunk_size, alloc_iter, free_chunk[free_index],
                                                       free_chunk[free_index + 1]);
                            t_weight_next_inserted =
                                get_weight(t_chunk, source_chunk_size, alloc_iter, free_chunk[free_index] - size(curr),
                                           free_chunk[free_index + 1]);
                        }


                        float t_weight_curr_inserted = 0.0f;

                        if (alloc_iter == t_allocs.begin())
                        {
                            t_weight_curr_inserted = get_target_weight(t_chunk, source_chunk_size, 0, 0,
                                                                       free_chunk[free_index] - size(curr));
                        }
                        else
                        {
                            t_weight_curr_inserted =
                                get_target_weight(t_chunk, source_chunk_size, end(std::prev(alloc_iter)), 0,
                                                  free_chunk[free_index] - size(curr));
                        }
                        float t_weight_before = t_weight_prev + t_weight_next;
                        float t_weight_after = t_weight_prev_inserted + t_weight_curr_inserted + t_weight_next_inserted;


                        float final_weight = (weight_after - weight_before) + (t_weight_after - t_weight_before);

                        if (final_weight < min_weight)
                        {
                            min_weight = final_weight;
                            FreeListEntry entry;

                            if (alloc_iter == t_allocs.cend())
                            {
                                // Allocate in last free space
                                entry = FreeListEntry{chunk.size - free_size, free_size};
                            }
                            else
                            {
                                entry = FreeListEntry{begin(alloc_iter) - free_size, free_size};
                            }
                            target = std::make_pair(t_chunk, entry);
                        }
                        targets_found++;
                    }

                    alloc_iter++;
                }
            }

            if (targets_found != 0)
            {
                int chunk  = std::get<0>(target);
                auto entry = std::get<1>(target);


                auto existing_defrag = std::find_if(defrag_operations.begin(), defrag_operations.end(),
                                                    [=](const DefragOperation& op) { return op.source == get(curr); });

                if (existing_defrag == defrag_operations.end())
                {
                    defrag_operations.insert(
                        DefragOperation{get(curr), chunks[chunk].chunk->memory, entry, min_weight});
                }
                else
                {
                    defrag_operations.erase(existing_defrag);
                    defrag_operations.insert(
                        DefragOperation{get(curr), chunks[chunk].chunk->memory, entry, min_weight});
                }
            }
        }
    }
}

template <typename T>
bool Defragger<T>::perform_defrag()
{
    auto op        = defrag_operations.begin();
    bool performed = false;
    while (running && op != defrag_operations.end())
    {
        if (allocator->memory_is_free(op->targetMemory, op->target) && op->source->is_dynamic())
        {
            op->source->mark_static();
            auto free_op = execute_copy_operation(*op);
            if (free_op)
            {
                SAIGA_ASSERT(free_op.value().target && free_op.value().source);
                allocator->swap(free_op.value().target, free_op.value().source);


                free_operations.push_back(free_op.value());
                performed = true;
            }
        }

        op = defrag_operations.erase(op);
    }
    return performed;
}


template <typename T>
float Defragger<T>::get_operation_penalty(ConstChunkIterator<T> target_chunk, ConstFreeIterator<T> target_location,
                                          ConstChunkIterator<T> source_chunk,
                                          ConstAllocationIterator<T> source_location) const
{
    auto& source_ptr = *source_location;
    float weight     = 0;


    if (target_chunk == source_chunk)
    {
        // move inside a chunk should be done after moving to others
        weight += penalties.same_chunk;
    }

    // if the move creates a hole that is smaller than the memory chunk itself -> add weight
    // anhand von der chunk size
    // ganz kleine löcher sind eigentlich auch egal -> smoothstep bei speicher / 1M (einfach 10kb)
    if (target_location->size != source_ptr->size && (target_location->size - source_ptr->size < source_ptr->size))
    {
        weight += penalties.target_small_hole * (1 - (static_cast<float>(source_ptr->size) / target_location->size));
    }

    // If move creates a hole at source -> add weight
    // auch wenn erste allokation
    // wenn kleine löcher davor oder danach auch smoothstep nehmen
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

    // minimales gewicht
    return weight;
}

template <typename T>
void Defragger<T>::apply_invalidations()
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

template <typename T>
void Defragger<T>::invalidate(vk::DeviceMemory memory)
{
    std::unique_lock<std::mutex> invalidate_lock(invalidate_mutex);
    invalidate_set.insert(memory);
}

template <typename T>
void Defragger<T>::invalidate(T* location)
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

    for (auto free_iter = free_operations.begin(); free_iter != free_operations.end();)
    {
        if (free_iter->source == location)
        {
            free_iter->source = nullptr;
        }
        else
        {
            ++free_iter;
        }
    }
}


template class Defragger<BufferMemoryLocation>;
template class Defragger<ImageMemoryLocation>;

std::optional<BufferDefragger::FreeOperation> BufferDefragger::execute_copy_operation(
    const BufferDefragger::DefragOperation& op)
{
    VLOG(1) << "DEFRAG" << *(op.source) << "->" << op.targetMemory << "," << op.target.offset << " " << op.target.size;

    BufferMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target, op.source->size);
    auto defrag_cmd                     = allocator->queue->commandPool.createAndBeginOneTimeBuffer();


    copy_buffer(defrag_cmd, reserve_space, op.source);

    defrag_cmd.end();

    allocator->queue->submitAndWait(defrag_cmd);

    allocator->queue->commandPool.freeCommandBuffer(defrag_cmd);

    // allocator->move_allocation(reserve_space, op.source);
    return std::optional<BufferDefragger::FreeOperation>(
        BufferDefragger::FreeOperation{reserve_space, op.source, frame_number + dealloc_delay});
}

std::optional<ImageDefragger::FreeOperation> ImageDefragger::execute_copy_operation(
    const ImageDefragger::DefragOperation& op)
{
    ImageMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target, op.source->size);

    auto new_data = op.source->data;
    new_data.create_image(device);


    bind_image_data(device, reserve_space, std::move(new_data));
    reserve_space->data.create_view(device);
    reserve_space->data.create_sampler(device);

    VLOG(1) << "IMAGE DEFRAG" << *(op.source) << "->" << *reserve_space;

    bool didcopy = img_copy_shader->copy_image(reserve_space, op.source);

    if (!didcopy)
    {
        return std::optional<ImageDefragger::FreeOperation>();
    }

    auto operation = std::optional<ImageDefragger::FreeOperation>(
        ImageDefragger::FreeOperation{reserve_space, op.source, frame_number + dealloc_delay});


    return operation;
}

ImageDefragger::ImageDefragger(VulkanBase* base, vk::Device device, BaseChunkAllocator<ImageMemoryLocation>* allocator,
                               uint32_t dealloc_delay, ImageCopyComputeShader* _img_copy_shader)
    : Defragger(base, device, allocator, dealloc_delay), img_copy_shader(_img_copy_shader)
{
    if (!img_copy_shader->is_initialized())
    {
        LOG(ERROR) << "Image copy shader could not be loaded. Image defragmentation is not possible.";
        valid = false;
    }
}

}  // namespace Saiga::Vulkan::Memory
