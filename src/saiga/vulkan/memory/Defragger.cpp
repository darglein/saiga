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
    uint64_t current_frame = base->current_frame;

    for (auto current = free_operations.begin(); running && current != free_operations.end();)
    {
        auto delay = current->delay;
        if (delay < current_frame)
        {
            if (current->target)
            {
                LOG(INFO) << "Free op " << *current;
                allocator->deallocate(current->target);
                current->source->mark_dynamic();
            }
            else
            {
                LOG(ERROR) << "Error free " << *current;
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
Defragger<T>::Defragger(VulkanBase* _base, vk::Device _device, BaseChunkAllocator<T>* _allocator,
                        uint32_t _dealloc_delay)
    : base(_base),
      dealloc_delay(_dealloc_delay + 15),
      // dealloc_delay(240),
      device(_device),
      allocator(_allocator),
      defrag_operations(),
      free_operations(),
      valid(true),
      enabled(false),
      running(false),
      quit(false),
      frame_number(0),
      worker(&Defragger::worker_func, this),
      queryPool(nullptr)
{
    auto queryPoolInfo = vk::QueryPoolCreateInfo{vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 2};
    queryPool          = base->device.createQueryPool(queryPoolInfo);
}

template <typename T>
void Defragger<T>::exit()
{
    if (quit)
    {
        return;
    }
    running = false;
    std::unique_lock<std::mutex> lock(running_mutex);

    {
        std::lock_guard<std::mutex> lock(start_mutex);
        running = false;
        quit    = true;
    }
    start_condition.notify_one();
    worker.join();

    if (queryPool)
    {
        base->device.destroyQueryPool(queryPool);
        queryPool = nullptr;
    }
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
        performed |= create_copy_commands();
        performed |= perform_free_operations();
    }
}


template <typename T>
void Defragger<T>::find_defrag_ops()
{
    // TODO: REFACTOR THIS!!!
    auto& chunks = allocator->chunks;

    // fill all free with free sizes.

    fill_free_list();

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

            // Get the current and after removal weight of the current allocation and its neighbours.
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

            auto targets_found = 0U;
            std::pair<size_t, FreeListEntry> target;
            // find up to config.max_targets targets and store the one with the highest delta weight
            for (auto t_chunk = 0U; targets_found < config.max_targets && t_chunk <= source_chunk_index; ++t_chunk)
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
                std::scoped_lock lock(defrag_mutex);
                auto chunk = std::get<0>(target);
                auto entry = std::get<1>(target);



                auto existing_defrag = std::find_if(defrag_operations.begin(), defrag_operations.end(),
                                                    [=](const DefragOperation& op) { return op.source == get(curr); });

                if (existing_defrag == defrag_operations.end())
                {
                    defrag_operations.push_back(
                        DefragOperation{get(curr), nullptr, chunks[chunk].chunk->memory, entry, min_weight, nullptr});
                    LOG(INFO) << "New op: " << defrag_operations.back();
                }
                else
                {
                    LOG(INFO) << "Old op" << *existing_defrag;
                }
                //                else if (!existing_defrag->copy_cmd && !existing_defrag->targetLocation)
                //                {
                //                    defrag_operations.erase(existing_defrag);
                //                    defrag_operations.push_back(
                //                        DefragOperation{get(curr), nullptr, chunks[chunk].chunk->memory, entry,
                //                        min_weight, nullptr});
                //                }
            }
        }
    }
    {
        std::scoped_lock lock(defrag_mutex);

        std::sort(defrag_operations.begin(), defrag_operations.end(),
                  [](const auto& entry, const auto& other) { return entry.weight < other.weight; });
    }
}

template <typename T>
void Defragger<T>::fill_free_list()
{
    auto& chunks = allocator->chunks;
    // contains a vector for each chunk, which in turn contains the free memory between the allocations. contains zeroes
    // if two allocations are next to each other.

    all_free.resize(chunks.size());
    for (auto i = 0U; running && i < chunks.size(); ++i)
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
}


template <typename T>
bool Defragger<T>::create_copy_commands()
{
    bool performed = false;

    std::scoped_lock defrag_lock(defrag_mutex);
    for (auto opIter = defrag_operations.begin(); running && opIter != defrag_operations.end(); ++opIter)
    {
        DefragOperation& op = *opIter;
        if (!op.copy_cmd && allocator->memory_is_free(op.targetMemory, op.target) && op.source->is_dynamic())
        {
            LOG(INFO) << "Cpy op" << op;
            auto cmd = base->mainQueue.commandPool.createAndBeginOneTimeBuffer();
            cmd.resetQueryPool(queryPool, 0, 2);
            cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, queryPool, 0);
            create_copy_command(op, cmd);
            cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, 1);
            cmd.end();
            op.copy_cmd = cmd;
            performed   = true;
        }
    }

    return performed;
}

template <typename T>
void Defragger<T>::perform_single_defrag(Defragger::DefragOperation& op)
{
    LOG(INFO) << "Run op " << op;
    op.source->mark_static();
    auto fence = base->mainQueue.submit(op.copy_cmd);

    auto result = vk::Result::eIncomplete;
    do
    {
        result = base->device.waitForFences(fence, VK_TRUE, 1000000000);

        if (result != vk::Result::eSuccess)
        {
            LOG(INFO) << vk::to_string(result);
        }
    } while (result != vk::Result::eSuccess);
    base->device.destroy(fence);
    base->mainQueue.commandPool.freeCommandBuffer(op.copy_cmd);
    std::array<uint64_t, 2> timestamps;
    base->device.getQueryPoolResults(queryPool, 0, 2, sizeof(uint64_t) * timestamps.size(), timestamps.data(), 8,
                                     vk::QueryResultFlagBits::e64);
    auto duration = timestamps[1] - timestamps[0];

    auto last = static_cast<double>(op.source->size / 1024) / duration;
    if (std::isinf(kbPerNanoSecond))
    {
        kbPerNanoSecond = last;
    }
    else
    {
        kbPerNanoSecond = 0.9 * last + 0.1 * kbPerNanoSecond;
    }

    auto free_op = FreeOperation{op.targetLocation, op.source, frame_number + dealloc_delay};

    allocator->swap(free_op.target, free_op.source);

    free_operations.push_back(free_op);
}
template <typename T>
int64_t Defragger<T>::perform_defrag(int64_t allowed_time)
{
    std::scoped_lock defrag_lock(defrag_mutex);
    auto remaining_time = allowed_time;

    bool first = true;
    auto op    = defrag_operations.begin();
    while (op != defrag_operations.end())
    {
        auto size = op->source->size;
        //        if (!first && (remaining_time - ((size / 1024) / kbPerNanoSecond)) < 0)
        //        {
        //            break;
        //        }
        if (op->copy_cmd && op->targetLocation)
        {
            perform_single_defrag(*op);
            remaining_time -= (size / 1024) / kbPerNanoSecond;
        }
        op    = defrag_operations.erase(op);
        first = false;
    }

    start();

    return remaining_time;
}


template class Defragger<BufferMemoryLocation>;
template class Defragger<ImageMemoryLocation>;


bool BufferDefragger::create_copy_command(Defragger::DefragOperation& op, vk::CommandBuffer cmd)
{
    BufferMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target, op.source->size);

    copy_buffer(cmd, reserve_space, op.source);

    op.targetLocation = reserve_space;
    return true;
}

bool ImageDefragger::create_copy_command(Defragger::DefragOperation& op, vk::CommandBuffer cmd)
{
    ImageMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target, op.source->size);

    auto new_data = op.source->data;
    new_data.create_image(device);

    bind_image_data(device, reserve_space, std::move(new_data));
    reserve_space->data.create_view(device);
    reserve_space->data.create_sampler(device);

    VLOG(1) << "IMAGE DEFRAG " << *(op.source) << "->" << *reserve_space;

    auto set = img_copy_shader->copy_image(cmd, reserve_space, op.source);

    if (!set)
    {
        return false;
    }

    usedSets.insert(std::make_pair(cmd, set.value()));

    op.targetLocation = reserve_space;
    return true;
}


ImageDefragger::ImageDefragger(VulkanBase* base, vk::Device device, BaseChunkAllocator<ImageMemoryLocation>* allocator,
                               uint32_t dealloc_delay, ImageCopyComputeShader* _img_copy_shader)
    : Defragger(base, device, allocator, dealloc_delay), usedSets(), img_copy_shader(_img_copy_shader)
{
    if (!img_copy_shader->is_initialized())
    {
        LOG(ERROR) << "Image copy shader could not be loaded. Image defragmentation is not possible.";
        valid = false;
    }
}

}  // namespace Saiga::Vulkan::Memory
