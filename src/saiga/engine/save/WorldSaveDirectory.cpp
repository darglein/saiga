/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The World's save directory members (docs/engine_design.md §15.1; S25): open, commit, wait, close, status.

#include "saiga/engine/core/Log.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/EntityUnloading.h"
#include "saiga/engine/save/SaveDirectory.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
SaveOpenResult World::openSaveDirectory(const std::string& path, const SaveDirectoryOptions& options)
{
    SaveOpenResult r;
    SAIGA_ASSERT(isHost(), "World::openSaveDirectory on a client (its state comes from the host)");
    SAIGA_ASSERT(!inStep() && commands.empty(), "World::openSaveDirectory inside a step");
    SAIGA_ASSERT(hookDepth == 0 && loadHookPhase == 0 && !loadInProgress,
                 "World::openSaveDirectory inside a game hook or a load");
    if (!isHost() || inStep() || hookDepth != 0 || loadHookPhase != 0 || loadInProgress)
    {
        r.error = "openSaveDirectory: only on the host, between steps and not in a hook";
        return r;
    }
    if (saveDir)
    {
        r.error = "openSaveDirectory: a save directory is open already (" + saveDir->status().path + ")";
        return r;
    }
    std::shared_ptr<SaveFileSystem> fs = options.fileSystem ? options.fileSystem : diskFileSystem(path, options.sync);
    // Installed before the open: the load's residency rebuild reads stored groups through it
    saveDir = std::make_unique<SaveDirectory>(*this, path, options, fs);
    r       = saveDir->open();
    if (!r.ok)
    {
        saveDir.reset();
        store->setTracking(false);
        SAIGA_LOG("save", "%s", r.error.c_str());
        return r;
    }
    SAIGA_LOG("save", "opened %s: commit %llu%s%s (%.1f ms, %llu records read)", fs->describe().c_str(),
              (unsigned long long)r.commit, r.created ? " (created)" : "", r.usedBackup ? " (the backup)" : "",
              r.openSeconds * 1000, (unsigned long long)r.recordsRead);
    return r;
}

bool World::hasSaveDirectory() const
{
    return saveDir != nullptr;
}

uint64_t World::commit()
{
    if (!isHost() || inStep() || hookDepth != 0 || loadHookPhase != 0 || !saveDir)
    {
        SAIGA_LOG("save", "commit refused: %s",
                  !saveDir ? "no save directory" : (!isHost() ? "a client" : "inside a step or a hook"));
        return 0;
    }
    return saveDir->commit();
}

bool World::waitForCommits()
{
    return saveDir ? saveDir->waitForCommits() : true;
}

bool World::closeSaveDirectory()
{
    if (!saveDir) return true;
    SAIGA_ASSERT(!inStep(), "World::closeSaveDirectory inside a step");
    const bool ok = saveDir->close();
    saveDir.reset();
    return ok;
}

SaveStatus World::saveStatus() const
{
    SaveStatus s         = saveDir ? saveDir->status() : SaveStatus();
    s.storedChunks       = store->chunksWithGroups();
    s.storedGroups       = store->totalGroups();
    s.unloadedSpaces     = store->unloadedSpaces().size();
    s.terrainMemoryBytes = terrainEditStore->memoryBytes();
    return s;
}

void World::afterStoreLoad(bool)
{
    unloading->onLoad();
    if (saveDir) saveDir->afterLoad();
}

}  // namespace Engine
}  // namespace Saiga
