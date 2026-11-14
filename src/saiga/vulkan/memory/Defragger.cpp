//
// Created by Peter Eichinger on 2019-01-21.
//

#include "Defragger.h"

#include "saiga/core/util/Thread/threadName.h"
#include "saiga/vulkan/Base.h"

#include "ImageCopyComputeShader.h"

#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
namespace Saiga::Vulkan::Memory
{
template <typename T>
Defragger<T>::Defragger(VulkanBase* _base, vk::Device _device, ChunkAllocator<T>* _allocator)
    : base(_base),
      device(_device),
      allocator(_allocator),
      valid(true),
      enabled(false),
      running(false),
      quit(false),
      kbPerNanoSecond(std::numeric_limits<double>::infinity())
{
    config.update_sizes(allocator->m_chunkSize);

    commandPool = base->mainQueue.createCommandPool(vk::CommandPoolCreateFlagBits::eTransient);

    vk::SemaphoreTypeCreateInfo timelineInfo(vk::SemaphoreType::eTimeline, 0);
    vk::SemaphoreCreateInfo semaphoreInfo;
    semaphoreInfo.pNext = &timelineInfo;
    copyTimeline        = device.createSemaphore(semaphoreInfo);

    auto family    = base->mainQueue.getQueueFamilyIndex();
    auto validBits = base->queueFamilyProperties.at(family).timestampValidBits;
    if (validBits > 0)
    {
        timestampMask   = validBits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << validBits) - 1);
        timestampPeriod = base->physicalDevice.getProperties().limits.timestampPeriod;
    }
    // The worker thread is started by the first start(), after the derived object is fully constructed.
}

template <typename T>
Defragger<T>::~Defragger()
{
    exit();
}

template <typename T>
void Defragger<T>::exit()
{
    if (exited)
    {
        return;
    }
    exited = true;

    {
        std::lock_guard<std::mutex> lock(start_mutex);
        running = false;
        quit    = true;
    }
    start_condition.notify_all();
    wait_condition.notify_all();
    if (worker.joinable())
    {
        worker.join();
    }

    // The submitted copies and the frames which used the old memory of the free ops must be complete.
    device.waitIdle();

    std::scoped_lock lock(defrag_mutex, copy_mutex);
    for (DefragOp& op : defragOps)
    {
        release_op_resources(op.cmd, op.queryPool);
        allocator->deallocate(op.target);
        op.source->mark_dynamic();
    }
    defragOps.clear();

    for (CopyOp& op : copyOps)
    {
        release_op_resources(op.cmd, op.queryPool);
        allocator->deallocate(op.target);
        op.source->mark_dynamic();
    }
    copyOps.clear();

    for (FreeOp& op : freeOps)
    {
        allocator->deallocate(op.target);
        if (op.source)
        {
            op.source->mark_dynamic();
        }
    }
    freeOps.clear();
    possibleOps.clear();
    currentDefragSources.clear();

    if (copyTimeline)
    {
        device.destroy(copyTimeline);
        copyTimeline = nullptr;
    }
    // The command pool is owned (and destroyed) by the queue that created it.
}

template <typename T>
void Defragger<T>::release_op_resources(vk::CommandBuffer cmd, vk::QueryPool queryPool)
{
    release_copy_resources(cmd);
    commandPool.freeCommandBuffer(cmd);
    if (queryPool)
    {
        device.destroy(queryPool);
    }
}

template <typename T>
vk::QueryPool Defragger<T>::create_query_pool()
{
    if (timestampMask == 0)
    {
        return nullptr;
    }
    return device.createQueryPool(vk::QueryPoolCreateInfo{vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 2});
}

template <typename T>
void Defragger<T>::start()
{
    if (!valid || !enabled || running || exited)
    {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        if (quit || pauseCount > 0)
        {
            return;
        }
        if (!worker.joinable())
        {
            worker = std::thread(&Defragger::worker_func, this);
        }
        running = true;
    }
    start_condition.notify_one();
}

template <typename T>
void Defragger<T>::pause()
{
    {
        // Under start_mutex: a concurrent start() either ran before (and is stopped below) or sees the count.
        std::lock_guard<std::mutex> lock(start_mutex);
        pauseCount++;
        running = false;
    }
    wait_condition.notify_all();
    {
        // Returns when the worker has left run().
        std::unique_lock<std::mutex> lock(running_mutex);
    }
}

template <typename T>
void Defragger<T>::resume()
{
    bool restart;
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        SAIGA_ASSERT(pauseCount > 0);
        pauseCount--;
        restart = pauseCount == 0;
    }
    if (restart) start();
}

template <typename T>
void Defragger<T>::stop()
{
    if (!valid)
    {
        return;
    }
    running = false;
    wait_condition.notify_all();
    {
        // Returns when the worker has left run().
        std::unique_lock<std::mutex> lock(running_mutex);
    }
}


template <typename T>
void Defragger<T>::worker_func()
{
    Saiga::setThreadName("Defragger");
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(start_mutex);
            start_condition.wait(lock, [this] { return running || quit; });
            if (quit)
            {
                return;
            }
        }
        std::unique_lock<std::mutex> running_lock(running_mutex);
        // stop() may have been called before this thread got the lock.
        if (!running)
        {
            continue;
        }
        if (allocator->chunks.empty())
        {
            // Nothing to do. Otherwise the wait predicate stays true and this thread spins.
            running = false;
            continue;
        }

        run();

        running = false;
    }
}

template <typename T>
void Defragger<T>::run()
{
    while (running)
    {
        bool progress = false;
        if (find_defrag_ops())
        {
            progress |= create_copy_commands();
        }
        progress |= complete_copy_commands();
        progress |= perform_free_operations(false);

        if (progress)
        {
            continue;
        }

        bool pending;
        {
            std::scoped_lock lock(defrag_mutex, copy_mutex);
            pending = !defragOps.empty() || !copyOps.empty() || !freeOps.empty();
        }
        if (!pending)
        {
            // Nothing can be moved anymore.
            break;
        }

        // Only waiting for the main thread (submission, frames) or the GPU.
        std::unique_lock<std::mutex> lock(wait_mutex);
        wait_condition.wait_for(lock, std::chrono::milliseconds(1));
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

            if (currentDefragSources.find(get(curr)) != currentDefragSources.end() || !is_movable(get(curr)))
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

                    // There is one more free slot than allocations: don't increment past the end.
                    if (alloc_iter != t_allocs.cend()) ++alloc_iter;
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
    bool created = false;

    std::scoped_lock defrag_lock(defrag_mutex);

    for (auto opIter = possibleOps.begin(); running && opIter != possibleOps.end(); opIter++)
    {
        PossibleOp& op = *opIter;

        auto existingOp = currentDefragSources.find(op.source);
        if (op.source->is_static() || existingOp != currentDefragSources.end() || !is_movable(op.source))
        {
            continue;
        }

        T* reserve_space =
            allocator->reserve_if_free(op.target.memory, static_cast<FreeListEntry>(op.target), op.source->size);

        if (!reserve_space)
        {
            continue;
        }

        op.source->mark_static();
        auto cmd       = commandPool.createAndBeginOneTimeBuffer();
        auto queryPool = create_query_pool();
        if (queryPool)
        {
            cmd.resetQueryPool(queryPool, 0, 2);
            cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, queryPool, 0);
        }
        bool recorded = create_copy_command(op, reserve_space, cmd);
        if (queryPool)
        {
            cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, queryPool, 1);
        }
        cmd.end();

        if (!recorded)
        {
            VLOG(3) << "Could not record the copy " << op;
            release_op_resources(cmd, queryPool);
            allocator->deallocate(reserve_space);
            op.source->mark_dynamic();
            continue;
        }

        auto [iterator, inserted] = currentDefragSources.insert(op.source);
        SAIGA_ASSERT(inserted && iterator != currentDefragSources.end(), "Source already in use");

        defragOps.push_back(DefragOp{reserve_space, op.source, cmd, queryPool});
        VLOG(3) << "Defrag " << defragOps.back();
        created = true;
    }

    possibleOps.clear();

    return created;
}


template <typename T>
std::pair<bool, int64_t> Defragger<T>::perform_defrag(int64_t allowed_time, vk::Semaphore frameTimeline,
                                                      uint64_t waitValue)
{
    std::pair<bool, int64_t> result;
    {
        std::scoped_lock defrag_lock(defrag_mutex);
        result = submit_defrag_ops(allowed_time, frameTimeline, waitValue);
    }

    if (result.first)
    {
        wait_condition.notify_all();
    }
    start();

    return result;
}

template <typename T>
std::pair<bool, int64_t> Defragger<T>::submit_defrag_ops(int64_t allowed_time, vk::Semaphore frameTimeline,
                                                         uint64_t waitValue)
{
    auto remaining_time = allowed_time;
    const double speed  = kbPerNanoSecond;

    bool first = true, performed = false;
    auto op = defragOps.begin();
    while (op != defragOps.end())
    {
        std::scoped_lock copy_lock(copy_mutex);

        // The first copy is always submitted, otherwise a copy that takes longer than the budget never happens.
        const double sizeInKB = static_cast<double>(op->source->size) / 1024.0;
        const auto duration   = std::isinf(speed) || speed <= 0 ? int64_t(0) : static_cast<int64_t>(sizeInKB / speed);
        if (!first && remaining_time - duration < 0)
        {
            break;
        }

        perform_single_defrag(*op, frameTimeline, waitValue);

        remaining_time -= duration;

        op        = defragOps.erase(op);
        first     = false;
        performed = true;
    }
    return std::make_pair(performed, remaining_time);
}


template <typename T>
void Defragger<T>::perform_single_defrag(Defragger<T>::DefragOp& op, vk::Semaphore frameTimeline, uint64_t waitValue)
{
    // The copy runs after the frame (frameTimeline >= waitValue) and after the previous copy (copyTimeline).
    std::array<vk::SemaphoreSubmitInfo, 2> waitSemaphores;
    uint32_t waitCount = 0;
    if (frameTimeline && waitValue > 0)
    {
        waitSemaphores[waitCount++] =
            vk::SemaphoreSubmitInfo(frameTimeline, waitValue, vk::PipelineStageFlagBits2::eAllCommands);
    }
    if (copyCounter > 0)
    {
        waitSemaphores[waitCount++] =
            vk::SemaphoreSubmitInfo(copyTimeline, copyCounter, vk::PipelineStageFlagBits2::eAllCommands);
    }

    const uint64_t signalValue = ++copyCounter;

    vk::CommandBufferSubmitInfo cmdInfo(op.cmd);
    vk::SemaphoreSubmitInfo signalInfo(copyTimeline, signalValue, vk::PipelineStageFlagBits2::eAllCommands);

    vk::SubmitInfo2 submit;
    submit.waitSemaphoreInfoCount   = waitCount;
    submit.pWaitSemaphoreInfos      = waitSemaphores.data();
    submit.commandBufferInfoCount   = 1;
    submit.pCommandBufferInfos      = &cmdInfo;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos    = &signalInfo;

    base->mainQueue.submit2(submit);

    copyOps.push_back(CopyOp{op.target, op.source, op.cmd, op.queryPool, signalValue});

    VLOG(3) << "Copy " << copyOps.back();
}

template <typename T>
void Defragger<T>::update_copy_speed(const CopyOp& copy)
{
    if (!copy.queryPool)
    {
        return;
    }
    std::array<uint64_t, 2> timestamps{};
    auto result = device.getQueryPoolResults(copy.queryPool, 0, 2, sizeof(uint64_t) * timestamps.size(),
                                             timestamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
    if (result != vk::Result::eSuccess)
    {
        return;
    }
    // Ticks -> ns. The mask handles a wrap around of the valid bits.
    const auto ticks      = (timestamps[1] - timestamps[0]) & timestampMask;
    const double duration = static_cast<double>(ticks) * timestampPeriod;
    if (duration <= 0)
    {
        return;
    }

    const double last = (static_cast<double>(copy.source->size) / 1024.0) / duration;
    VLOG(3) << "DURATION: " << duration << " ns " << last << " KB/ns";
    const double current = kbPerNanoSecond;
    if (std::isinf(current))
    {
        kbPerNanoSecond = last;
    }
    else
    {
        kbPerNanoSecond = 0.9 * last + 0.1 * current;
    }
}

template <typename T>
bool Defragger<T>::complete_copy_commands()
{
    bool completed = false;
    while (true)
    {
        // copyOps is filled by the main thread, so it must only be accessed under the lock.
        std::scoped_lock copy_lock(copy_mutex);
        if (copyOps.empty())
        {
            break;
        }

        CopyOp& copy = copyOps.front();

        if (device.getSemaphoreCounterValue(copyTimeline) < copy.signal_value)
        {
            break;
        }

        update_copy_speed(copy);

        allocator->swap(copy.target, copy.source);

        // The frame that is recorded now may still use the old memory (now in copy.target).
        freeOps.push_back(FreeOp{copy.target, copy.source, base->currentFrameSignalValue()});

        VLOG(3) << "Free " << freeOps.back();
        copy.source->modified();

        release_op_resources(copy.cmd, copy.queryPool);

        copyOps.pop_front();
        completed = true;
    }
    return completed;
}

template <typename T>
bool Defragger<T>::perform_free_operations(bool force)
{
    if (freeOps.empty())
    {
        return false;
    }
    const uint64_t completed_value = force ? std::numeric_limits<uint64_t>::max() : base->completedFrameValue();

    bool performed = false;
    for (auto current = freeOps.begin(); (running || force) && current != freeOps.end();)
    {
        if (current->wait_value <= completed_value)
        {
            VLOG(3) << "Free op " << *current;
            allocator->deallocate(current->target);
            if (current->source)
            {
                current->source->mark_dynamic();
            }
            auto found = currentDefragSources.find(current->source);
            SAIGA_ASSERT(found != currentDefragSources.end());
            currentDefragSources.erase(found);
            current   = freeOps.erase(current);
            performed = true;
        }
        else
        {
            current++;
        }
    }
    return performed;
}

template <typename T>
void Defragger<T>::full_defrag()
{
    if (!valid || exited)
    {
        return;
    }

    // The worker must not run concurrently. Stop it first: while it waits in run() for the main thread (which is
    // blocked here), it holds running_mutex, so locking without stopping would deadlock. The ops it already recorded
    // are submitted and completed below. It blocks on this lock if it is started meanwhile.
    stop();
    std::unique_lock<std::mutex> running_lock(running_mutex);
    running = true;

    while (running)
    {
        bool created = false;
        if (find_defrag_ops())
        {
            created = create_copy_commands();
        }

        // Submit all copies (also the ones recorded by the worker before). They wait for the submitted frames.
        const uint64_t submittedFrames = base->current_frame;
        {
            std::scoped_lock defrag_lock(defrag_mutex);
            submit_defrag_ops(std::numeric_limits<int64_t>::max(), base->frameTimeline, submittedFrames);
        }

        uint64_t lastCopy;
        {
            std::scoped_lock copy_lock(copy_mutex);
            lastCopy = copyCounter;
        }
        if (lastCopy > 0)
        {
            vk::SemaphoreWaitInfo waitInfo;
            waitInfo.semaphoreCount = 1;
            waitInfo.pSemaphores    = &copyTimeline;
            waitInfo.pValues        = &lastCopy;
            auto result             = device.waitSemaphores(waitInfo, std::numeric_limits<uint64_t>::max());
            SAIGA_ASSERT(result == vk::Result::eSuccess);
        }
        complete_copy_commands();

        // The old memory may be used by the frames in flight. No frame is recorded while this function blocks.
        base->waitFrameValue(submittedFrames);
        perform_free_operations(true);

        if (!created)
        {
            break;
        }
    }
    running = false;
}

template <typename T>
void Defragger<T>::invalidate(T* memoryLocation)
{
    SAIGA_ASSERT(!running, "invalidate() may only be called when defragger is stopped");

    auto defrag               = currentDefragSources.find(memoryLocation);
    auto isCurrentlyDefragged = defrag != currentDefragSources.end();

    if (!isCurrentlyDefragged)
    {
        return;
    }

    VLOG(3) << "in use " << PointerOutput<T>(memoryLocation);
    std::scoped_lock lock(defrag_mutex, copy_mutex);

    auto defragOp = std::find_if(defragOps.begin(), defragOps.end(),
                                 [&](const auto& entry) { return entry.source == memoryLocation; });

    if (defragOp != defragOps.end())
    {
        release_op_resources(defragOp->cmd, defragOp->queryPool);
        allocator->deallocate(defragOp->target);
        defragOps.erase(defragOp);
    }

    auto copyOp =
        std::find_if(copyOps.begin(), copyOps.end(), [&](const auto& entry) { return entry.source == memoryLocation; });

    if (copyOp != copyOps.end())
    {
        // Later copies wait for smaller timeline values, so they are not affected by dropping this one.
        vk::SemaphoreWaitInfo waitInfo;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores    = &copyTimeline;
        waitInfo.pValues        = &copyOp->signal_value;
        auto result             = device.waitSemaphores(waitInfo, std::numeric_limits<uint64_t>::max());
        SAIGA_ASSERT(result == vk::Result::eSuccess);

        release_op_resources(copyOp->cmd, copyOp->queryPool);
        allocator->deallocate(copyOp->target);
        copyOps.erase(copyOp);
    }

    auto freeOp =
        std::find_if(freeOps.begin(), freeOps.end(), [&](const auto& entry) { return entry.source == memoryLocation; });
    if (freeOp != freeOps.end())
    {
        allocator->deallocate(freeOp->target);
        freeOps.erase(freeOp);
    }

    currentDefragSources.erase(defrag);
}



template class Defragger<BufferMemoryLocation>;
template class Defragger<ImageMemoryLocation>;


BufferDefragger::~BufferDefragger()
{
    exit();
}

bool BufferDefragger::create_copy_command(BufferDefragger::PossibleOp& op, BufferMemoryLocation* reserve_space,
                                          vk::CommandBuffer cmd)

{
    VLOG(3) << "Create Buffer Copy" << PointerOutput<BufferMemoryLocation>(reserve_space) << " " << op << " " << cmd;
    copy_buffer(cmd, reserve_space, op.source);

    // Later frames read the moved data (vertex/index/uniform/storage) after the swap; make the copy's writes
    // available and visible to them.
    vk::MemoryBarrier2 barrier(vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite,
                               vk::PipelineStageFlagBits2::eAllCommands,
                               vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, barrier, nullptr, nullptr));
    return true;
}

ImageDefragger::ImageDefragger(VulkanBase* base, vk::Device device, ChunkAllocator<ImageMemoryLocation>* allocator,
                               ImageCopyComputeShader* _img_copy_shader)
    : Defragger(base, device, allocator), usedSets(), img_copy_shader(_img_copy_shader)
{
    if (!img_copy_shader->is_initialized())
    {
        LOG(ERROR) << "Image copy shader could not be loaded. Image defragmentation is not possible.";
        valid = false;
    }
}

ImageDefragger::~ImageDefragger()
{
    exit();
}

bool ImageDefragger::is_movable(ImageMemoryLocation* location)
{
    return ImageCopyComputeShader::can_copy(location->data);
}

bool ImageDefragger::create_copy_command(ImageDefragger::PossibleOp& op, ImageMemoryLocation* reserve_space,
                                         vk::CommandBuffer cmd)
{
    auto new_data = op.source->data;
    // The copies of the handles must not be destroyed by new_data.
    new_data.image   = nullptr;
    new_data.view    = nullptr;
    new_data.sampler = nullptr;
    new_data.create_image(device);

    bind_image_data(device, reserve_space, std::move(new_data));
    reserve_space->data.create_view(device);
    reserve_space->data.create_sampler(device);

    auto set = img_copy_shader->copy_image(cmd, reserve_space, op.source);
    if (!set)
    {
        // The image, view and sampler are destroyed by the deallocation of reserve_space.
        return false;
    }
    std::scoped_lock lock(sets_mutex);
    usedSets.insert(std::make_pair(cmd, set.value()));
    return true;
}

void ImageDefragger::release_copy_resources(vk::CommandBuffer cmd)
{
    std::scoped_lock lock(sets_mutex);
    auto found = usedSets.find(cmd);
    if (found != usedSets.end())
    {
        // The copy is complete or was never submitted.
        base->descriptorPool.freeDescriptorSet(found->second);
        usedSets.erase(found);
    }
}

}  // namespace Saiga::Vulkan::Memory
