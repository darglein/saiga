//
// Created by Peter Eichinger on 2019-01-21.
//

#include "Defragger.h"

#include "saiga/core/util/Thread/threadName.h"
#include "saiga/vulkan/Base.h"

#include "ImageCopyComputeShader.h"

#include <optional>
namespace Saiga::Vulkan::Memory
{
template <typename T>
bool Defragger<T>::perform_free_operations()
{
    uint64_t current_frame = base->current_frame;

    for (auto current = freeOps.begin(); running && current != freeOps.end();)
    {
        auto delay = current->delay;
        if (delay < current_frame)
        {
            if (current->target)
            {
                VLOG(3) << "Free op " << *current;
                allocator->deallocate(current->target);
                if (current->source)
                {
                    current->source->mark_dynamic();
                }
            }
            else
            {
                LOG(ERROR) << "Error free " << *current;
            }
            auto found = currentDefragSources.find(current->source);
            SAIGA_ASSERT(found != currentDefragSources.end());
            currentDefragSources.erase(found);
            current = freeOps.erase(current);
        }
        else
        {
            current++;
        }
    }
    return !freeOps.empty();
}

template <typename T>
Defragger<T>::Defragger(VulkanBase* _base, vk::Device _device, ChunkAllocator<T>* _allocator, uint32_t _dealloc_delay)
    : base(_base),
      dealloc_delay(_dealloc_delay),
      // dealloc_delay(240),
      device(_device),
      allocator(_allocator),
      possibleOps(),
      defragOps(),
      copyOps(),
      freeOps(),
      currentDefragSources(),
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
    config.update_sizes(allocator->m_chunkSize);

    commandPool = base->mainQueue.createCommandPool(vk::CommandPoolCreateFlagBits::eTransient);
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
    for (CopyOp& copyOp : copyOps)
    {
        base->device.waitForFences(copyOp.fence, VK_TRUE, std::numeric_limits<uint64_t>::max());

        base->device.destroy(copyOp.fence);
        base->device.destroy(copyOp.signal_semaphore);
        commandPool.freeCommandBuffer(copyOp.cmd);
    }

    for (DefragOp& defragOp : defragOps)
    {
        commandPool.freeCommandBuffer(defragOp.cmd);
    }
    //    commandPool.destroy();
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
    while (running)
    {
        performed = false;
        performed |= find_defrag_ops();
        performed |= create_copy_commands();
        performed |= complete_copy_commands();
        performed |= perform_free_operations();

        if (!performed)
        {
            running = false;
        }
    }
}


template <typename T>
bool Defragger<T>::find_defrag_ops()
{
    bool performed = false;

    auto& chunks = allocator->chunks;

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

            if (currentDefragSources.find(get(curr)) != currentDefragSources.end())
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
            Target target;
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
                            Target entry;

                            if (alloc_iter == t_allocs.cend())
                            {
                                // Allocate in last free space
                                entry = Target{chunk.memory, chunk.size - free_size, free_size};
                            }
                            else
                            {
                                entry = Target{chunk.memory, begin(alloc_iter) - free_size, free_size};
                            }
                            target = entry;
                        }
                        targets_found++;
                    }

                    alloc_iter++;
                }
            }

            if (targets_found != 0)
            {
                std::scoped_lock lock(defrag_mutex);

                auto* current = get(curr);
                //                auto [iterator, inserted] = currentDefragSources.insert(current);

                //                SAIGA_ASSERT(inserted, "There was already a ");

                auto found = std::find_if(possibleOps.begin(), possibleOps.end(),
                                          [=](const PossibleOp& op) { return op.source == current; });

                if (found != possibleOps.end())
                {
                    *found = PossibleOp{target, current, min_weight};
                }
                else
                {
                    possibleOps.push_back(PossibleOp{target, current, min_weight});
                }

                performed = true;
            }
        }
    }

    if (performed)
    {
        std::sort(possibleOps.begin(), possibleOps.end(),
                  [](const auto& entry, const auto& other) { return entry.weight < other.weight; });
    }
    return performed;

}  // namespace Saiga::Vulkan::Memory

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
    //    bool performed = false;

    std::scoped_lock defrag_lock(defrag_mutex);

    for (auto opIter = possibleOps.begin(); running && opIter != possibleOps.end(); opIter++)
    {
        PossibleOp& op = *opIter;

        auto existingOp = currentDefragSources.find(op.source);
        if (op.source->is_static() || existingOp != currentDefragSources.end())
        {
            continue;
        }

        T* reserve_space =
            allocator->reserve_if_free(op.target.memory, static_cast<FreeListEntry>(op.target), op.source->size);
        ;
        if (reserve_space)
        {
            //            VLOG(3) << "/*Cpy*/ op" << op;
            op.source->mark_static();
            auto cmd = commandPool.createAndBeginOneTimeBuffer();
            cmd.resetQueryPool(queryPool, 0, 2);
            cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, queryPool, 0);
            create_copy_command(op, reserve_space, cmd);
            cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, 1);
            cmd.end();

            auto [iterator, inserted] = currentDefragSources.insert(op.source);
            SAIGA_ASSERT(inserted && iterator != currentDefragSources.end(), "Source already in use");

            defragOps.push_back(DefragOp{reserve_space, op.source, cmd});
            VLOG(3) << "Defrag " << defragOps.back();
        }
        //        performed = true;
    }

    possibleOps.clear();

    return !defragOps.empty();
}


template <typename T>
std::pair<bool, int64_t> Defragger<T>::perform_defrag(int64_t allowed_time, vk::Semaphore semaphore)
{
    std::scoped_lock defrag_lock(defrag_mutex);
    auto remaining_time = allowed_time;

    bool first = true, performed = false;
    auto op = defragOps.begin();
    while (op != defragOps.end())
    {
        std::scoped_lock copy_lock(copy_mutex);

        auto sizeInKB = op->source->size / 1024;
        if (!first && (remaining_time - (sizeInKB / kbPerNanoSecond)) < 0)
        {
            break;
        }

        perform_single_defrag(*op, first ? semaphore : nullptr);

        remaining_time -= sizeInKB / kbPerNanoSecond;

        op        = defragOps.erase(op);
        first     = false;
        performed = true;
    }

    start();

    return std::make_pair(performed, remaining_time);
}


template <typename T>
void Defragger<T>::perform_single_defrag(Defragger<T>::DefragOp& op, vk::Semaphore semaphore)
{
    //    std::scoped_lock copy_lock(copy_mutex);
    //    VLOG(3) << "Run op " << op;

    vk::SubmitInfo submit;

    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &op.cmd;
    vk::Semaphore wait;
    std::array<vk::PipelineStageFlags, 2> waitStage{
        vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eAllGraphics,
        vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eAllGraphics};

    std::vector<vk::Semaphore> waitSemaphores;

    if (semaphore)
    {
        waitSemaphores.push_back(semaphore);
    }
    if (copyOps.empty())
    {
        wait = nullptr;
    }
    else
    {
        waitSemaphores.push_back(copyOps.back().signal_semaphore);
        wait = copyOps.back().signal_semaphore;
    }

    auto fence        = device.createFence(vk::FenceCreateInfo{});
    auto sigSemaphore = device.createSemaphore(vk::SemaphoreCreateInfo{});


    submit.waitSemaphoreCount   = waitSemaphores.size();
    submit.pWaitSemaphores      = waitSemaphores.data();
    submit.pWaitDstStageMask    = waitStage.data();
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &sigSemaphore;

    base->mainQueue.submit(submit, fence);


    copyOps.push_back(CopyOp{op.target, op.source, op.cmd, fence, wait, sigSemaphore});

    VLOG(3) << "Copy " << copyOps.back();
}

template <typename T>
bool Defragger<T>::complete_copy_commands()
{
    while (!copyOps.empty())
    {
        {
            using namespace std::chrono_literals;
            //            std::this_thread::sleep_for(10us);
        }
        {
            std::scoped_lock copy_lock(copy_mutex);

            CopyOp& copy = copyOps.front();

            auto result = device.waitForFences(copy.fence, VK_TRUE, 1000);
            if (result != vk::Result::eSuccess)
            {
                break;
            }

            std::array<uint64_t, 2> timestamps;
            base->device.getQueryPoolResults(queryPool, 0, 2, sizeof(uint64_t) * timestamps.size(), timestamps.data(),
                                             8, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
            auto duration = timestamps[1] - timestamps[0];

            auto last = static_cast<double>(copy.source->size / 1024) / duration;
            VLOG(3) << "DURATION: " << duration << " " << last;
            if (std::isinf(kbPerNanoSecond))
            {
                kbPerNanoSecond = last;
            }
            else
            {
                kbPerNanoSecond = 0.9 * last + 0.1 * kbPerNanoSecond;
            }

            allocator->swap(copy.target, copy.source);

            freeOps.push_back(FreeOp{copy.target, copy.source, frame_number + dealloc_delay});

            VLOG(3) << "Free " << freeOps.back();
            copy.source->modified();


            device.destroy(copy.fence);

            if (copy.wait_semaphore)
            {
                device.destroy(copy.wait_semaphore);
            }
            if (copy.signal_semaphore && copyOps.size() == 1)
            {
                device.destroy(copy.signal_semaphore);
            }
            commandPool.freeCommandBuffer(copy.cmd);

            copyOps.pop_front();
        }
    }
    return !copyOps.empty();
}

template <typename T>
void Defragger<T>::invalidate(T* memoryLocation)
{
    SAIGA_ASSERT(!running, "invalidate() may only be called when defragger is stopped");

    auto defrag               = currentDefragSources.find(memoryLocation);
    auto isCurrentlyDefragged = defrag != currentDefragSources.end();

    if (isCurrentlyDefragged)
    {
        VLOG(3) << "in use " << PointerOutput<T>(memoryLocation);
        std::scoped_lock lock(defrag_mutex, copy_mutex);

        auto defragOp = std::find_if(defragOps.begin(), defragOps.end(),
                                     [&](const auto& entry) { return entry.source == memoryLocation; });

        if (defragOp != defragOps.end())
        {
            commandPool.freeCommandBuffer(defragOp->cmd);
            allocator->deallocate(defragOp->target);
            defragOps.erase(defragOp);
        }

        auto copyOp = std::find_if(copyOps.begin(), copyOps.end(),
                                   [&](const auto& entry) { return entry.source == memoryLocation; });

        if (copyOp != copyOps.end())
        {
            base->device.waitForFences(copyOp->fence, VK_TRUE, std::numeric_limits<uint64_t>::max());

            base->device.destroy(copyOp->fence);
            if (copyOp->wait_semaphore)
            {
                base->device.destroy(copyOp->wait_semaphore);
            }
            if (copyOp != copyOps.begin())
            {
                std::prev(copyOp)->signal_semaphore = nullptr;
            }

            commandPool.freeCommandBuffer(copyOp->cmd);
            allocator->deallocate(copyOp->target);
            copyOps.erase(copyOp);
        }

        auto freeOp = std::find_if(freeOps.begin(), freeOps.end(),
                                   [&](const auto& entry) { return entry.source == memoryLocation; });
        if (freeOp != freeOps.end())
        {
            freeOp->source = nullptr;

            allocator->deallocate(freeOp->target);

            freeOps.erase(freeOp);
        }

        currentDefragSources.erase(defrag);
    }
}



template class Defragger<BufferMemoryLocation>;
template class Defragger<ImageMemoryLocation>;


void BufferDefragger::create_copy_command(BufferDefragger::PossibleOp& op, BufferMemoryLocation* reserve_space,
                                          vk::CommandBuffer cmd)

{
    //    BufferMemoryLocation* reserve_space = allocator->reserve_space(op.targetMemory, op.target,
    //    op.source->size);
    VLOG(3) << "Create Buffer Copy" << PointerOutput<BufferMemoryLocation>(reserve_space) << " " << op << " " << cmd;
    copy_buffer(cmd, reserve_space, op.source);
}

void ImageDefragger::create_copy_command(ImageDefragger::PossibleOp& op, ImageMemoryLocation* reserve_space,
                                         vk::CommandBuffer cmd)
{
    auto new_data = op.source->data;
    new_data.create_image(device);

    bind_image_data(device, reserve_space, std::move(new_data));
    reserve_space->data.create_view(device);
    reserve_space->data.create_sampler(device);

    auto set = img_copy_shader->copy_image(cmd, reserve_space, op.source);
    usedSets.insert(std::make_pair(cmd, set.value()));
}


ImageDefragger::ImageDefragger(VulkanBase* base, vk::Device device, ChunkAllocator<ImageMemoryLocation>* allocator,
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
