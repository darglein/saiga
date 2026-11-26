/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// MemorySaveFileSystem (docs/engine_design.md §15.1; S25, G9): a save directory in memory with a crash model, for tests
// and ScenarioWorld. Files are nodes with their current and their durable bytes (syncFile makes the current durable);
// directories have current and durable entries (create, rename and remove change the current entries; syncDirectory
// makes them durable). Every mutating call is counted and logged; failAfter(n) makes the calls after the n-th fail
// without a change (with tornWrite the first failing append / writeNew applies the first half of its bytes). The
// images of a crash are new file systems: afterKill (a killed process: the current state), afterPowerLoss (only durable
// entries and durable bytes) and afterEntriesOnly (the current entries with the durable bytes: a rename that reached
// the disk before the renamed file's data). Reads and lock are not counted. Thread safe (one mutex).

#pragma once

#include "saiga/engine/save/SaveFileSystem.h"

#include <map>
#include <mutex>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API MemorySaveFileSystem final : public SaveFileSystem
{
   public:
    enum class OpKind : uint8_t
    {
        MakeDirectory,
        WriteNew,
        Append,
        Truncate,
        SyncFile,
        SyncDirectory,
        Rename,
        Remove
    };
    struct Op
    {
        OpKind kind = OpKind::WriteNew;
        std::string path, to;
        std::vector<uint8_t> data;
        uint64_t size = 0;
        std::string describe() const;
    };

    MemorySaveFileSystem();
    ~MemorySaveFileSystem() override;

    bool exists(const std::string& path) override;
    bool list(const std::string& dir, std::vector<std::string>& names, std::string* error) override;
    bool isDirectory(const std::string& path) override;
    bool fileSize(const std::string& path, uint64_t& size, std::string* error) override;
    bool readAt(const std::string& path, uint64_t offset, uint64_t size, uint8_t* out, std::string* error) override;
    bool makeDirectory(const std::string& dir, std::string* error) override;
    bool writeNew(const std::string& path, const uint8_t* data, size_t size, std::string* error) override;
    bool append(const std::string& path, const uint8_t* data, size_t size, std::string* error) override;
    bool truncate(const std::string& path, uint64_t size, std::string* error) override;
    bool syncFile(const std::string& path, std::string* error) override;
    bool syncDirectory(const std::string& dir, std::string* error) override;
    bool rename(const std::string& from, const std::string& to, std::string* error) override;
    bool remove(const std::string& path, std::string* error) override;
    bool lock(std::string* error) override;
    void unlock() override;
    std::string describe() const override { return "memory"; }

    // The mutating calls so far (also the failed ones) and their log
    uint64_t operations() const;
    std::vector<Op> operationLog() const;
    void clearLog();
    // The calls after the n-th mutating one (counted from now) fail with "simulated crash" (-1: never)
    void failAfter(int64_t n, bool tornWrite = false);
    // The whole state (not the lock, the counter or the log)
    std::shared_ptr<MemorySaveFileSystem> clone() const;
    // A clone with the first n operations of log applied (torn: the (n+1)-th, if it is an append or writeNew, with the
    // first half of its bytes)
    std::shared_ptr<MemorySaveFileSystem> replay(const std::vector<Op>& log, size_t n, bool torn) const;
    std::shared_ptr<MemorySaveFileSystem> afterKill() const { return clone(); }
    std::shared_ptr<MemorySaveFileSystem> afterPowerLoss() const;
    std::shared_ptr<MemorySaveFileSystem> afterEntriesOnly() const;
    // A hash of the visible state (current entries and bytes): equal images are opened once
    uint64_t stateHash() const;
    // Every file path (recursive, sorted)
    std::vector<std::string> files() const;
    // Test helpers (not counted): write / read a whole file directly (current and durable)
    void putFile(const std::string& path, const std::vector<uint8_t>& data);
    std::vector<uint8_t> fileBytes(const std::string& path) const;

   private:
    struct Node
    {
        bool dir = false;
        std::vector<uint8_t> data, durable;
        std::map<std::string, int> entries, durableEntries;
    };
    mutable std::mutex mutex;
    std::vector<Node> nodes;  // node 0: the directory itself (always durable)
    std::vector<Op> log;
    uint64_t ops      = 0;
    int64_t failLimit = -1;
    bool torn         = false;
    bool tornUsed     = false;
    bool locked       = false;

    // The node of a path in the current entries (-1: none); the parent node and the last name
    int resolve(const std::string& path) const;
    bool parentOf(const std::string& path, int& parent, std::string& name) const;
    // Counts the call; false (simulated crash) past the limit
    bool count(const Op& op, std::string* error, bool* tornNow);
    bool apply(const Op& op, std::string* error, bool tornHalf);
    void copyTree(const MemorySaveFileSystem& from, int fromNode, int toNode, bool durableEntries, bool durableBytes);
};

}  // namespace Engine
}  // namespace Saiga
