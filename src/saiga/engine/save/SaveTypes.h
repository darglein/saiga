/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The options, status and open result of a save directory (docs/engine_design.md §15.1; S25): World::openSaveDirectory,
// commit, saveStatus.

#pragma once

#include "saiga/engine/save/LoadResult.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace Saiga
{
namespace Engine
{
class SaveFileSystem;

struct SaveDirectoryOptions
{
    // null: diskFileSystem(path, sync)
    std::shared_ptr<SaveFileSystem> fileSystem;
    // World time between automatic commits (game policy, §21); 0: only commit()
    double commitIntervalSeconds = 300;
    // syncFile / syncDirectory of the disk file system (off only for tests on a real disk; the memory file system
    // models them either way)
    bool sync     = true;
    int zstdLevel = 3;
    // The commit's IO runs inside commit(), reads inside the step (tests, ScenarioWorld); unset: follows
    // streamingSettings().synchronous
    std::optional<bool> synchronousIo;
};

struct SaveStatus
{
    bool open = false;
    std::string path;
    uint64_t committed = 0, inFlight = 0;  // the newest successful commit; the commit being written (0: none)
    uint64_t lastCommitStep = 0;
    double lastCommitTime   = 0;  // world time of the last freeze
    std::string lastError;        // the last failed commit's error (empty after a success)
    uint64_t dirtyChunks = 0, storedChunks = 0, storedGroups = 0, unloadedSpaces = 0;
    uint64_t regionFiles = 0, regionBytes = 0, liveBytes = 0;
    uint64_t terrainMemoryBytes = 0;  // TerrainEditStore::memoryBytes (G2: the modification state stays in memory)
};

struct SaveOpenResult
{
    bool ok = false;
    std::string error;         // set when !ok: the world is unchanged
    bool created    = false;   // the directory had no commit: the current world was adopted and committed
    uint64_t commit = 0;       // the loaded (or first) commit
    bool usedBackup = false;   // the newest world-state file was corrupt
    LoadResult load;           // the world-state load (hook errors as for tryLoad)
    double openSeconds   = 0;  // wall time of the open
    uint64_t recordsRead = 0;  // records read for the modification state
};

}  // namespace Engine
}  // namespace Saiga
