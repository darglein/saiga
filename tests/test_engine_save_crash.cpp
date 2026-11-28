/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The crash enumeration of a save directory's commits (docs/engine_design.md §15.1; S25, G9): each case runs its world
// once to commit c, changes the world, records the reference blobs (save() at c and at the freeze of c + 1), then
// commits c + 1 on a MemorySaveFileSystem that logs its K mutating operations. Every image is built from that run by
// replaying a prefix of the log onto a clone of the pre-commit state (n = 0..K; four kinds: a killed process, a killed
// process with a torn last write, directory entries without their data, a power loss), opened in a fresh world and
// checked: the open succeeds, it loads c or c + 1, monotone in n per kind, c + 1 at n = K for every kind (the protocol
// syncs before its commit point), the loaded world saves the reference blob of its commit (so every PersistentId
// exists exactly once: resident or in a stored group), and at n = 0, K / 2 and K no .tmp or .corrupt file remains
// and one more commit and open succeed. Equal images are opened once (a hash of the file system state).

#include "saiga/engine/core/Log.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/MemorySaveFileSystem.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/save/SaveDirectory.h"
#include "saiga/engine/terrain/TerrainBrush.h"

#include "gtest/gtest.h"

#include <functional>
#include <set>

#include "unload_test_setup.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace UnloadTest;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

const Vec3 kA(1000, 0, 1000), kB(5000, 0, 1000);

TerrainBrush addBrush(double x, double z)
{
    TerrainBrush b;
    b.op       = TerrainBrush::Add;
    b.center   = Vec3(x, 0, z);
    b.radius   = 8;
    b.strength = 2;
    return b;
}

SaveDirectoryOptions memOptions(std::shared_ptr<SaveFileSystem> fs)
{
    SaveDirectoryOptions o;
    o.fileSystem            = std::move(fs);
    o.synchronousIo         = true;
    o.zstdLevel             = 1;
    o.commitIntervalSeconds = 0;
    return o;
}

void teleport(World& w, Entity e, const Vec3& p)
{
    w.get<Transform>(e).teleport(p, Quat::Identity());
}

struct CrashRun
{
    std::unique_ptr<World> w;
    Entity player;
    std::shared_ptr<MemorySaveFileSystem> fs;
};

// The world of every case: the player at A, three Growers at B (unloaded in the second step), an edit at A
CrashRun makeRun()
{
    CrashRun r;
    r.w = std::make_unique<World>(unloadTestSetup());
    r.w->setStreamingSynchronous(true);
    r.player = addPlayer(*r.w, kB);
    for (int i = 0; i < 3; ++i) r.w->makePersistent(r.w->spawn<Grower>(kB + Vec3(20.0 * i, 0, 0)));
    r.w->step();
    teleport(*r.w, r.player, kA);
    r.w->editTerrain(0, addBrush(1100, 1100));
    r.w->step();
    r.fs = std::make_shared<MemorySaveFileSystem>();
    return r;
}

enum class Image
{
    Kill,
    KillTorn,
    EntriesOnly,
    PowerLoss
};

// Opens the image in a fresh world; returns the loaded commit (0: failed, with a test failure)
uint64_t openImage(std::shared_ptr<MemorySaveFileSystem> fs, std::vector<uint8_t>* blob, bool* created = nullptr)
{
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    const SaveOpenResult r = v.openSaveDirectory("mem", memOptions(fs));
    EXPECT_TRUE(r.ok) << r.error;
    if (!r.ok) return 0;
    if (created) *created = r.created;
    if (blob) *blob = v.save();
    v.closeSaveDirectory();
    return r.commit;
}

// No temporary or corrupt file is left after the open's cleanup (stray region files are covered by the next commit and
// open succeeding, which they would break)
void expectClean(const MemorySaveFileSystem& fs, const std::string& where)
{
    for (const std::string& f : fs.files())
    {
        EXPECT_EQ(f.find(".tmp"), std::string::npos) << where << ": " << f;
        EXPECT_EQ(f.find(".corrupt"), std::string::npos) << where << ": " << f;
    }
}

// The enumeration: before() runs to commit c (the directory is open), change() then changes the world; the commit of
// c + 1 is enumerated. firstCommit: the open (adoption) itself is enumerated
void enumerate(const std::function<void(CrashRun&)>& before, const std::function<void(CrashRun&)>& change,
               bool firstCommit = false)
{
    CrashRun r = makeRun();
    std::vector<uint8_t> blobC;
    uint64_t c = 0;
    std::shared_ptr<MemorySaveFileSystem> base;
    if (firstCommit)
    {
        change(r);
        base = r.fs->clone();
        r.fs->clearLog();
        const SaveOpenResult o = r.w->openSaveDirectory("mem", memOptions(r.fs));
        ASSERT_TRUE(o.ok) << o.error;
    }
    else
    {
        ASSERT_TRUE(r.w->openSaveDirectory("mem", memOptions(r.fs)).ok);
        before(r);
        c     = r.w->saveStatus().committed;
        blobC = r.w->save();
        change(r);
        base = r.fs->clone();
        r.fs->clearLog();
        ASSERT_EQ(r.w->commit(), c + 1);
        ASSERT_TRUE(r.w->saveStatus().lastError.empty());
    }
    const std::vector<uint8_t> blobC1 = r.w->save();
    const auto log                    = r.fs->operationLog();
    const size_t K                    = log.size();
    ASSERT_GT(K, 3u);
    std::set<uint64_t> opened;
    for (Image kind : {Image::Kill, Image::KillTorn, Image::EntriesOnly, Image::PowerLoss})
    {
        bool reachedNew = false;
        for (size_t n = 0; n <= K; ++n)
        {
            std::shared_ptr<MemorySaveFileSystem> img = base->replay(log, n, kind == Image::KillTorn);
            if (kind == Image::EntriesOnly) img = img->afterEntriesOnly();
            if (kind == Image::PowerLoss) img = img->afterPowerLoss();
            const bool check = n == 0 || n == K / 2 || n == K;
            if (!opened.insert(img->stateHash()).second && !check && !reachedNew) continue;
            const std::string where = "image " + std::to_string(int(kind)) + " at operation " + std::to_string(n) +
                                      " of " + std::to_string(K) + (n < K ? " (" + log[n].describe() + ")" : "");
            std::vector<uint8_t> blob;
            bool created          = false;
            const uint64_t loaded = openImage(img, &blob, &created);
            if (::testing::Test::HasFailure()) FAIL() << where;
            if (firstCommit)
            {
                // The open either adopts (a fresh world: created) or finds commit 1
                EXPECT_TRUE(created || loaded == 1) << where;
                if (!created)
                {
                    EXPECT_EQ(blob, blobC1) << where;
                }
                if (loaded == 1 && !created) reachedNew = true;
            }
            else
            {
                ASSERT_TRUE(loaded == c || loaded == c + 1) << where << ": loaded " << loaded;
                // Monotone: once an image of a kind loads c + 1, every later one does
                if (reachedNew)
                {
                    EXPECT_EQ(loaded, c + 1) << where;
                }
                reachedNew = reachedNew || loaded == c + 1;
                EXPECT_EQ(blob, loaded == c ? blobC : blobC1) << where;
            }
            if (n == K)
            {
                // Every operation replayed: the commit survives (a first commit is never adopted again)
                EXPECT_EQ(loaded, c + 1) << where;
                EXPECT_FALSE(created) << where;
            }
            if (check)
            {
                // The open cleaned the image: no .tmp or corrupt file; one more commit and open succeed
                expectClean(*img, where);
                World v(unloadTestSetup());
                v.setStreamingSynchronous(true);
                ASSERT_TRUE(v.openSaveDirectory("mem", memOptions(img)).ok) << where;
                v.editTerrain(0, addBrush(1600, 1600));
                const uint64_t next = v.commit();
                EXPECT_NE(next, 0u) << where;
                const std::vector<uint8_t> after = v.save();
                v.closeSaveDirectory();
                std::vector<uint8_t> again;
                EXPECT_EQ(openImage(img, &again), next) << where;
                EXPECT_TRUE(again == after) << where << ": " << [&]
                {
                    DiffFilter f;
                    f.limit          = 5;
                    const SaveDiff d = diffSaves(after, again, f);
                    std::string t    = d.error;
                    for (const SaveDifference& x : d.differences) t += "\n  " + x.describe();
                    return t;
                }();
            }
        }
        if (!firstCommit)
        {
            EXPECT_TRUE(reachedNew);
        }
    }
}
}  // namespace

TEST(SaveCrash, KillDuringCommit)
{
    // Commit c + 1 with edited chunks and resident entities
    enumerate([](CrashRun&) {},
              [](CrashRun& r)
              {
                  r.w->editTerrain(0, addBrush(1400, 1100));
                  r.w->spawn<Grower>(kA + Vec3(10, 0, 10));
                  r.w->step();
              });
}

TEST(SaveCrash, UnloadBetweenCommits)
{
    // The acceptance: after commit c the player leaves; a group is unloaded (in memory only): commit c has it resident
    // in world.c.sgw, commit c + 1 in its record
    enumerate(
        [](CrashRun& r)
        {
            r.w->makePersistent(r.w->spawn<Grower>(kA + Vec3(30, 0, 0)));
            r.w->step();
            r.w->commit();
        },
        [](CrashRun& r)
        {
            teleport(*r.w, r.player, kB);
            r.w->step();
        });
}

TEST(SaveCrash, ReloadBetweenCommits)
{
    // Unloaded before commit c (in its record, bytes dropped), reloaded after it: never duplicated
    enumerate([](CrashRun&) {},
              [](CrashRun& r)
              {
                  teleport(*r.w, r.player, kB);
                  r.w->step();
              });
}

TEST(SaveCrash, FirstCommit)
{
    // The directory's creation at every point: the open either adopts or finds commit 1
    enumerate([](CrashRun&) {}, [](CrashRun&) {}, true);
}

TEST(SaveCrash, CompactingCommit)
{
    // A large group rewritten by every commit until the next one compacts its region file
    enumerate(
        [](CrashRun& r)
        {
            teleport(*r.w, r.player, kB);
            r.w->step();
            const Entity root = r.w->spawn<Grower>(kB + Vec3(10, 0, 0));
            Rng rng(5);
            for (int i = 0; i < 10000; ++i)
            {
                const Entity e = r.w->create();
                r.w->add<Parent>(e, Parent{root});
                r.w->get<Transform>(e).setScale(Vec3(rng.uniform(), rng.uniform(), rng.uniform()));
            }
            r.w->step();
            teleport(*r.w, r.player, kA);
            r.w->step();
            r.w->commit();
            // Commits until the region file reaches the threshold: the next commit compacts it
            for (int i = 0; i < 40; ++i)
            {
                const ChunkEntry* e = r.w->chunkStoreView().entry(chunkOf(0, kB.x(), kB.z()));
                uint64_t size       = 0;
                ASSERT_TRUE(e && e->record);
                r.fs->fileSize(e->record->file.path(), size, nullptr);
                if (size >= kCompactMinBytes) break;
                r.w->editTerrain(0, addBrush(kB.x() + 60, 900 + i));
                r.w->commit();
            }
        },
        [](CrashRun& r)
        {
            r.w->editTerrain(0, addBrush(kB.x() + 70, 920));
            ASSERT_EQ(r.w->chunkStoreView().stats().compactions, 0u);
        });
}

TEST(SaveCrash, HeaderRewrite)
{
    // A commit that appends a schema table: a group written with another table (a Crate on a reloaded root is unloaded
    // again with the current table after a table of another build was interned)
    enumerate(
        [](CrashRun& r)
        {
            auto other = std::make_shared<SchemaTable>();
            other->types.push_back(SchemaType{SchemaKind::Component, "Other", 200, 1, {}});
            other->buildIndex();
            // A table of another build in the store (as a directory of an older build would carry): the current one is
            // appended by the next unload
            r.w->chunkStore().internTable(other);
        },
        [](CrashRun& r)
        {
            r.w->spawn<Grower>(kA + Vec3(30, 0, 0));
            r.w->step();
            teleport(*r.w, r.player, kB);
            r.w->step();
        });
}

TEST(SaveCrash, KillDuringOpenCleanup)
{
    // A directory with garbage (a .tmp file, a corrupt newest commit, a stray region file); the open's cleanup killed
    // at every point: the next open loads the same commit
    CrashRun r = makeRun();
    ASSERT_TRUE(r.w->openSaveDirectory("mem", memOptions(r.fs)).ok);
    r.w->editTerrain(0, addBrush(1400, 1100));
    ASSERT_EQ(r.w->commit(), 2u);
    r.w->editTerrain(0, addBrush(1500, 1100));
    ASSERT_EQ(r.w->commit(), 3u);
    r.w->closeSaveDirectory();
    std::vector<uint8_t> world3 = r.fs->fileBytes("world.3.sgw");
    world3[30] ^= 0x55;
    r.fs->putFile("world.3.sgw", world3);
    r.fs->putFile("world.9.sgw.tmp", {1, 2});
    r.fs->putFile("regions/0/7.7.0.sgr", {1, 2, 3});
    const auto base = r.fs->clone();
    std::vector<uint8_t> reference;
    {
        auto img = base->clone();
        img->clearLog();
        World v(unloadTestSetup());
        v.setStreamingSynchronous(true);
        const SaveOpenResult o = v.openSaveDirectory("mem", memOptions(img));
        ASSERT_TRUE(o.ok) << o.error;
        ASSERT_EQ(o.commit, 2u);
        reference = v.save();
        v.closeSaveDirectory();
        const auto log = img->operationLog();
        ASSERT_GT(log.size(), 2u);
        for (size_t n = 0; n <= log.size(); ++n)
            for (bool power : {false, true})
            {
                auto image = base->replay(log, n, false);
                if (power) image = image->afterPowerLoss();
                std::vector<uint8_t> blob;
                EXPECT_EQ(openImage(image, &blob), 2u) << n;
                EXPECT_EQ(blob, reference) << n;
            }
    }
}
