/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The save directory (docs/engine_design.md §15.1; S25): the file system layer (the memory file system's crash model,
// the disk file system), create and open, commits and retention, corrupt files and the backup, the refusals of an open
// over a save, commits and loads with the IO lane held, region generations, unreadable records, schema tables and lazy
// migration, compaction, asynchronous versus synchronous IO, failed commits, the records of every generated chunk, the
// lock, and the committed directory (R-SAVE-1). Memory file system unless named; flat worlds of
// tests/unload_test_setup.h with terrain edits and stored groups.

#include "saiga/core/util/string_to_path.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/save/ChunkRecordCodec.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/MemorySaveFileSystem.h"
#include "saiga/engine/save/SaveDirectory.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/terrain/TerrainBrush.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/saiga_buildconfig.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "streaming_test_setup.h"
#include "unload_test_setup.h"
#include <condition_variable>

using namespace Saiga;
using namespace Saiga::Engine;
using namespace UnloadTest;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// A and B in different regions (16 chunks: 4096 m)
const Vec3 kA(1000, 0, 1000), kB(5000, 0, 1000);

std::unique_ptr<World> makeWorld(const WorldSetup& s)
{
    auto w = std::make_unique<World>(s);
    w->setStreamingSynchronous(true);
    return w;
}

void teleport(World& w, Entity e, const Vec3& p)
{
    w.get<Transform>(e).teleport(p, Quat::Identity());
}

TerrainBrush addBrush(double x, double z, double radius = 8, double strength = 2)
{
    TerrainBrush b;
    b.op       = TerrainBrush::Add;
    b.center   = Vec3(x, 0, z);
    b.radius   = radius;
    b.strength = strength;
    return b;
}

SaveDirectoryOptions memOptions(std::shared_ptr<SaveFileSystem> fs, bool synchronous = true)
{
    SaveDirectoryOptions o;
    o.fileSystem            = std::move(fs);
    o.synchronousIo         = synchronous;
    o.zstdLevel             = 1;
    o.commitIntervalSeconds = 0;
    return o;
}

// A world with edits at A, stored groups at B (unloaded in the second step) and the player at A
struct Scene
{
    std::unique_ptr<World> w;
    Entity player;
    std::vector<PersistentId> stored;
};
Scene makeScene(const WorldSetup& s = unloadTestSetup())
{
    Scene sc;
    sc.w      = makeWorld(s);
    World& w  = *sc.w;
    sc.player = addPlayer(w, kB);
    for (int i = 0; i < 3; ++i) sc.stored.push_back(w.makePersistent(w.spawn<Grower>(kB + Vec3(20.0 * i, 0, 0))));
    w.step();
    teleport(w, sc.player, kA);
    w.editTerrain(0, addBrush(kA.x() + 100, kA.z() + 100));
    w.editTerrain(0, addBrush(1152, 1152 + 1024));
    w.step();
    return sc;
}

void flipByte(MemorySaveFileSystem& fs, const std::string& path, size_t offset)
{
    std::vector<uint8_t> b = fs.fileBytes(path);
    ASSERT_LT(offset, b.size());
    b[offset] ^= 0x5A;
    fs.putFile(path, b);
}

bool hasFile(const MemorySaveFileSystem& fs, const std::string& path)
{
    const std::vector<std::string> f = fs.files();
    return std::find(f.begin(), f.end(), path) != f.end();
}

// ---- a latch for the IO lane (the hook of a task never outlives it)
struct Latch
{
    std::mutex m;
    std::condition_variable cv;
    bool closed = false;
    int entered = 0;
    void close()
    {
        std::lock_guard<std::mutex> l(m);
        closed  = true;
        entered = 0;
    }
    void open()
    {
        {
            std::lock_guard<std::mutex> l(m);
            closed = false;
        }
        cv.notify_all();
    }
    void wait()
    {
        std::unique_lock<std::mutex> l(m);
        if (closed) entered++;
        cv.notify_all();
        cv.wait(l, [this] { return !closed; });
    }
    void waitEntered(int n)
    {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return entered >= n || !closed; });
    }
};
Latch& latch()
{
    static Latch l;
    return l;
}
}  // namespace

// ==== SaveFileSystem =================================================================================================

TEST(SaveFileSystem, MemoryCrashModel)
{
    MemorySaveFileSystem fs;
    const std::vector<uint8_t> a{1, 2, 3, 4}, b{5, 6};
    ASSERT_TRUE(fs.writeNew("x", a.data(), a.size(), nullptr));
    // Not synced: gone after a power loss, present after a kill
    EXPECT_FALSE(fs.afterPowerLoss()->exists("x"));
    EXPECT_TRUE(fs.afterKill()->exists("x"));
    ASSERT_TRUE(fs.syncFile("x", nullptr));
    // The entry is not durable yet: entries-only keeps it (its bytes durable), power loss drops it
    EXPECT_FALSE(fs.afterPowerLoss()->exists("x"));
    EXPECT_EQ(fs.afterEntriesOnly()->fileBytes("x"), a);
    ASSERT_TRUE(fs.syncDirectory("", nullptr));
    EXPECT_EQ(fs.afterPowerLoss()->fileBytes("x"), a);
    // A rename is atomic in both images and durable only after syncDirectory
    ASSERT_TRUE(fs.writeNew("y.tmp", b.data(), b.size(), nullptr));
    ASSERT_TRUE(fs.syncFile("y.tmp", nullptr));
    ASSERT_TRUE(fs.rename("y.tmp", "x", nullptr));
    EXPECT_EQ(fs.afterKill()->fileBytes("x"), b);
    EXPECT_EQ(fs.afterPowerLoss()->fileBytes("x"), a);
    ASSERT_TRUE(fs.syncDirectory("", nullptr));
    EXPECT_EQ(fs.afterPowerLoss()->fileBytes("x"), b);
    // An append without a sync: entries-only shows the durable bytes
    ASSERT_TRUE(fs.append("x", a.data(), a.size(), nullptr));
    EXPECT_EQ(fs.afterEntriesOnly()->fileBytes("x"), b);
    // failAfter(n): the (n+1)-th mutating call and every later one fail without a change; reads are not counted
    const uint64_t ops = fs.operations();
    fs.failAfter(1);
    EXPECT_TRUE(fs.append("x", b.data(), b.size(), nullptr));
    std::string e;
    EXPECT_FALSE(fs.writeNew("z", a.data(), a.size(), &e));
    EXPECT_FALSE(fs.exists("z"));
    EXPECT_FALSE(fs.remove("x", nullptr));
    uint64_t size = 0;
    EXPECT_TRUE(fs.fileSize("x", size, nullptr));
    EXPECT_EQ(size, 8u);
    EXPECT_EQ(fs.operations(), ops + 3);
    // A torn append leaves exactly its first half
    fs.failAfter(0, true);
    const std::vector<uint8_t> six{9, 9, 9, 9, 9, 9};
    EXPECT_FALSE(fs.append("x", six.data(), six.size(), nullptr));
    EXPECT_TRUE(fs.fileSize("x", size, nullptr));
    EXPECT_EQ(size, 11u);
    fs.failAfter(-1);
    // Replay of the log onto a clone of the empty state
    MemorySaveFileSystem empty;
    const auto log = fs.operationLog();
    auto image     = empty.replay(log, 2, false);
    EXPECT_EQ(image->fileBytes("x"), a);
}

TEST(SaveFileSystem, DiskRoundTrip)
{
    const std::string dir =
        path_to_string(std::filesystem::temp_directory_path() /
                       ("saiga_s25_disk_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()))));
    std::filesystem::remove_all(string_to_path(dir));
    std::filesystem::create_directories(string_to_path(dir));
    auto fs = diskFileSystem(dir);
    std::string e;
    const std::vector<uint8_t> a{1, 2, 3, 4};
    // A UTF-8 file name
    const std::string name = "w\xc3\xb6rld.bin";
    ASSERT_TRUE(fs->append(name, a.data(), a.size(), &e)) << e;
    ASSERT_TRUE(fs->append(name, a.data(), a.size(), &e)) << e;
    std::vector<uint8_t> out(4);
    ASSERT_TRUE(fs->readAt(name, 4, 4, out.data(), &e)) << e;
    EXPECT_EQ(out, a);
    ASSERT_TRUE(fs->truncate(name, 3, &e)) << e;
    uint64_t size = 0;
    ASSERT_TRUE(fs->fileSize(name, size, &e));
    EXPECT_EQ(size, 3u);
    ASSERT_TRUE(fs->syncFile(name, &e)) << e;
    ASSERT_TRUE(fs->writeNew("other", a.data(), 2, &e)) << e;
    ASSERT_TRUE(fs->rename("other", name, &e)) << e;
    ASSERT_TRUE(fs->syncDirectory("", &e)) << e;
    ASSERT_TRUE(fs->makeDirectory("sub", &e)) << e;
    std::vector<std::string> names;
    ASSERT_TRUE(fs->list("", names, &e));
    EXPECT_EQ(names, (std::vector<std::string>{"sub", name}));
    ASSERT_TRUE(fs->fileSize(name, size, &e));
    EXPECT_EQ(size, 2u);
    ASSERT_TRUE(fs->remove(name, &e));
    EXPECT_FALSE(fs->exists(name));
    // The lock: a second lock of the same directory fails
    ASSERT_TRUE(fs->lock(&e)) << e;
    auto second = diskFileSystem(dir);
    EXPECT_FALSE(second->lock(&e));
    fs->unlock();
    EXPECT_TRUE(second->lock(&e)) << e;
    second->unlock();
    std::filesystem::remove_all(string_to_path(dir));
}

// ==== SaveDirectory ==================================================================================================

TEST(SaveDirectory, CreateAndOpen)
{
    resetHooks();
    Scene sc = makeScene();
    World& w = *sc.w;
    ASSERT_EQ(w.chunkStoreView().totalGroups(), 3u);
    auto fs                = std::make_shared<MemorySaveFileSystem>();
    const SaveOpenResult r = w.openSaveDirectory("mem", memOptions(fs));
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_TRUE(r.created);
    EXPECT_EQ(r.commit, 1u);
    EXPECT_TRUE(hasFile(*fs, "lock"));
    EXPECT_TRUE(hasFile(*fs, "header.sgh"));
    EXPECT_TRUE(hasFile(*fs, "world.1.sgw"));
    // The groups' chunk is not resident: their bytes left memory (save() reads them back)
    EXPECT_FALSE(w.chunkStoreView().needsRead(chunkOf(0, kA.x(), kA.z())));
    const std::vector<uint8_t> blob = w.save();
    EXPECT_TRUE(w.closeSaveDirectory());
    EXPECT_EQ(w.save(), blob);
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    const SaveOpenResult o = v.openSaveDirectory("mem", memOptions(fs));
    ASSERT_TRUE(o.ok) << o.error;
    EXPECT_FALSE(o.created);
    EXPECT_EQ(o.commit, 1u);
    // The records read: the chunks with deltas and the space record of the unloaded cellar (no centre)
    size_t withDeltas = 0;
    for (const auto& [c, e] : w.terrainEdits().chunks(0)) withDeltas += e.nonZeroSamples > 0 ? 1 : 0;
    EXPECT_FALSE(v.spaceLoaded(1));
    EXPECT_EQ(o.recordsRead, withDeltas + 1);
    EXPECT_EQ(v.save(), blob);
    EXPECT_EQ(v.hash(), w.hash());
    // The stored groups are on disk only; they reload when the player returns
    EXPECT_TRUE(v.chunkStoreView().needsRead(chunkOf(0, kB.x(), kB.z())));
    teleport(v, sc.player, kB);
    v.step();
    for (PersistentId id : sc.stored) EXPECT_TRUE(v.resolve(id).valid());
}

TEST(SaveDirectory, CommitsAndRetention)
{
    Scene sc = makeScene();
    World& w = *sc.w;
    auto fs  = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
    const uint64_t written = w.chunkStoreView().stats().recordsWritten;
    EXPECT_EQ(w.commit(), 2u);
    // An unchanged world commits no record
    EXPECT_EQ(w.chunkStoreView().stats().recordsWritten, written);
    // An edit inside one chunk writes exactly one record
    w.editTerrain(0, addBrush(1152 + 256, 1152));
    EXPECT_EQ(w.commit(), 3u);
    EXPECT_EQ(w.chunkStoreView().stats().recordsWritten, written + 1);
    EXPECT_FALSE(hasFile(*fs, "world.1.sgw"));
    EXPECT_TRUE(hasFile(*fs, "world.2.sgw"));
    EXPECT_TRUE(hasFile(*fs, "world.3.sgw"));
    const std::vector<uint8_t> blob = w.save();
    w.closeSaveDirectory();
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    const SaveOpenResult o = v.openSaveDirectory("mem", memOptions(fs));
    ASSERT_TRUE(o.ok) << o.error;
    EXPECT_EQ(o.commit, 3u);
    EXPECT_EQ(v.save(), blob);
}

TEST(SaveDirectory, CorruptNewestUsesBackup)
{
    Scene sc = makeScene();
    World& w = *sc.w;
    auto fs  = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
    w.editTerrain(0, addBrush(1152 + 256, 1152));
    ASSERT_EQ(w.commit(), 2u);
    const std::vector<uint8_t> blob2 = w.save();
    w.editTerrain(0, addBrush(1152 + 512, 1152));
    ASSERT_EQ(w.commit(), 3u);
    w.closeSaveDirectory();
    flipByte(*fs, "world.3.sgw", 40);
    {
        World v(unloadTestSetup());
        v.setStreamingSynchronous(true);
        const SaveOpenResult o = v.openSaveDirectory("mem", memOptions(fs));
        ASSERT_TRUE(o.ok) << o.error;
        EXPECT_EQ(o.commit, 2u);
        EXPECT_TRUE(o.usedBackup);
        EXPECT_TRUE(hasFile(*fs, "world.3.sgw.corrupt"));
        EXPECT_EQ(v.save(), blob2);
    }
    // Both corrupt (the .corrupt file and a corrupt commit 2): an error, nothing renamed or removed, the world
    // unchanged
    flipByte(*fs, "world.2.sgw", 40);
    const std::vector<std::string> before = fs->files();
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    const std::vector<uint8_t> fresh = v.save();
    const SaveOpenResult o           = v.openSaveDirectory("mem", memOptions(fs));
    EXPECT_FALSE(o.ok);
    EXPECT_EQ(fs->files(), before);
    EXPECT_EQ(v.save(), fresh);
    EXPECT_FALSE(v.hasSaveDirectory());
}

TEST(SaveDirectory, NeverAdoptsOverASave)
{
    auto tryOpen = [](std::shared_ptr<MemorySaveFileSystem> fs)
    {
        World v(unloadTestSetup());
        return v.openSaveDirectory("mem", memOptions(fs));
    };
    // The files other than the lock file (the open creates it before it looks at the directory)
    auto files = [](const MemorySaveFileSystem& fs)
    {
        std::vector<std::string> f = fs.files();
        f.erase(std::remove(f.begin(), f.end(), std::string("lock")), f.end());
        return f;
    };
    const std::vector<uint8_t> junk{1, 2, 3};
    {
        auto fs = std::make_shared<MemorySaveFileSystem>();
        fs->putFile("world.1.sgw", junk);
        fs->putFile("header.sgh", junk);
        const auto before = files(*fs);
        EXPECT_FALSE(tryOpen(fs).ok);
        EXPECT_EQ(files(*fs), before);
    }
    {
        auto fs = std::make_shared<MemorySaveFileSystem>();
        fs->putFile("world.4.sgw.corrupt", junk);
        EXPECT_FALSE(tryOpen(fs).ok);
        EXPECT_FALSE(hasFile(*fs, "world.1.sgw"));
    }
    {
        auto fs = std::make_shared<MemorySaveFileSystem>();
        fs->putFile("notes.txt", junk);
        const SaveOpenResult r = tryOpen(fs);
        EXPECT_FALSE(r.ok);
        EXPECT_NE(r.error.find("notes.txt"), std::string::npos);
    }
    {
        // The garbage of a crashed first commit is adopted; the stale region file is recreated
        auto fs = std::make_shared<MemorySaveFileSystem>();
        fs->putFile("lock", {});
        fs->putFile("world.1.sgw.tmp", junk);
        fs->putFile("header.sgh", junk);
        fs->putFile("regions/0/0.0.0.sgr", junk);
        Scene sc               = makeScene();
        const SaveOpenResult r = sc.w->openSaveDirectory("mem", memOptions(fs));
        ASSERT_TRUE(r.ok) << r.error;
        EXPECT_TRUE(r.created);
        EXPECT_FALSE(hasFile(*fs, "world.1.sgw.tmp"));
        EXPECT_NE(fs->fileBytes("regions/0/0.0.0.sgr"), junk);
    }
}

TEST(SaveDirectory, LoadWhileCommitInFlight)
{
    Scene sc = makeScene();
    World& w = *sc.w;
    auto fs  = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs, false)).ok);
    ASSERT_TRUE(w.waitForCommits());
    const std::vector<uint8_t> blob           = w.save();
    w.saveDirectory()->ioQueue().hookForTests = [] { latch().wait(); };
    latch().close();
    w.editTerrain(0, addBrush(1152 + 256, 1152));
    ASSERT_EQ(w.commit(), 2u);
    EXPECT_EQ(w.saveStatus().inFlight, 2u);
    // The load waits for the commit (held: the latch opens once the task waits on it)
    std::thread opener(
        []
        {
            latch().waitEntered(1);
            latch().open();
        });
    ASSERT_TRUE(w.tryLoad(blob).ok());
    opener.join();
    w.saveDirectory()->ioQueue().hookForTests = nullptr;
    EXPECT_EQ(w.saveStatus().inFlight, 0u);
    EXPECT_EQ(w.saveStatus().committed, 2u);
    EXPECT_EQ(w.save(), blob);
    // The loaded store is written by the next commit
    ASSERT_EQ(w.commit(), 3u);
    ASSERT_TRUE(w.waitForCommits());
    w.closeSaveDirectory();
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    ASSERT_TRUE(v.openSaveDirectory("mem", memOptions(fs)).ok);
    EXPECT_EQ(v.save(), blob);
}

TEST(SaveDirectory, RegionGenerationNeverReused)
{
    Scene sc = makeScene();
    World& w = *sc.w;
    auto fs  = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
    // The groups (region (1, 0) of B: chunk 19) reload: the chunk is empty, its region drops out of the newest index
    const std::string g0 = "regions/0/1.0.0.sgr";
    ASSERT_TRUE(hasFile(*fs, g0));
    teleport(w, sc.player, kB);
    w.step();
    ASSERT_EQ(w.commit(), 2u);
    EXPECT_TRUE(hasFile(*fs, g0));
    // A new chunk with state in that region: a new generation (the backup names generation 0)
    w.editTerrain(0, addBrush(4600, 1152));
    ASSERT_EQ(w.commit(), 3u);
    EXPECT_TRUE(hasFile(*fs, "regions/0/1.0.1.sgr"));
    w.closeSaveDirectory();
    // Commit 3 is corrupt: the backup opens
    flipByte(*fs, "world.3.sgw", 40);
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    const SaveOpenResult o = v.openSaveDirectory("mem", memOptions(fs));
    ASSERT_TRUE(o.ok) << o.error;
    EXPECT_EQ(o.commit, 2u);
}

TEST(SaveDirectory, RecordErrors)
{
    // Unit cases of the record decoder: an unknown section tag, a section version above 1, a frame claiming 300 MB
    {
        RecordContent rc;
        rc.key              = chunkOf(0, 0, 0);
        rc.commit           = 1;
        rc.hasRecord        = true;
        rc.placementVersion = 1;
        std::vector<uint8_t> bytes;
        encodeRecord(rc, 1, bytes);
        RecordContent out;
        std::string e;
        ASSERT_TRUE(decodeRecord(bytes.data(), bytes.size(), out, e)) << e;
        auto withPayload = [&](const std::vector<uint8_t>& payload)
        {
            std::vector<uint8_t> frame;
            zstdCompress(payload.data(), payload.size(), 1, frame);
            std::vector<uint8_t> r(bytes.begin(), bytes.begin() + kRecordHeaderSize);
            const uint32_t n = uint32_t(frame.size());
            for (int i = 0; i < 4; ++i) r[28 + size_t(i)] = uint8_t(n >> (8 * i));
            r.insert(r.end(), frame.begin(), frame.end());
            const uint64_t h = fnv1a64Bytes(r.data(), r.size());
            for (int i = 0; i < 8; ++i) r.push_back(uint8_t(h >> (8 * i)));
            return r;
        };
        auto section = [](const char(&tag)[5], uint32_t version)
        {
            std::vector<uint8_t> p;
            BinaryWriter w(p);
            w.write(saveTag(tag));
            w.write(version);
            w.write(uint64_t(0));
            return p;
        };
        std::vector<uint8_t> bad = withPayload(section("ZZZZ", 1));
        EXPECT_FALSE(decodeRecord(bad.data(), bad.size(), out, e));
        EXPECT_NE(e.find("ZZZZ"), std::string::npos);
        EXPECT_NE(e.find("chunk (0, 0, 0)"), std::string::npos);
        bad = withPayload(section("RREC", 2));
        EXPECT_FALSE(decodeRecord(bad.data(), bad.size(), out, e));
        EXPECT_NE(e.find("RREC"), std::string::npos);
        // A frame header claiming 300 MB: refused before decompressing
        std::vector<uint8_t> frame{0x28, 0xB5, 0x2F, 0xFD, 0xE0};
        const uint64_t big = uint64_t(300) << 20;
        for (int i = 0; i < 8; ++i) frame.push_back(uint8_t(big >> (8 * i)));
        frame.insert(frame.end(), {0, 0, 0});
        std::vector<uint8_t> decompressed;
        EXPECT_FALSE(zstdDecompress(frame.data(), frame.size(), decompressed, e));
        EXPECT_NE(e.find("256 MB"), std::string::npos);
    }
    // A flipped payload byte in a record with groups: the open succeeds (groups are read at reload), the reload logs
    // and counts them; an edit of the chunk then commits a record without them
    Scene sc = makeScene();
    World& w = *sc.w;
    auto fs  = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
    const ChunkEntry* e = w.chunkStoreView().entry(chunkOf(0, kB.x(), kB.z()));
    ASSERT_TRUE(e && e->record);
    const RecordLocation loc = *e->record;
    w.closeSaveDirectory();
    flipByte(*fs, loc.file.path(), size_t(loc.offset) + kRecordHeaderSize + 4);
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    ASSERT_TRUE(v.openSaveDirectory("mem", memOptions(fs)).ok);
    teleport(v, sc.player, kB);
    v.step();
    EXPECT_EQ(v.chunkStoreView().stats().failedGroups, 3u);
    for (PersistentId id : sc.stored) EXPECT_FALSE(v.resolve(id).valid());
    v.editTerrain(0, addBrush(kB.x() + 60, 900));
    ASSERT_EQ(v.commit(), 2u);
    EXPECT_TRUE(v.saveStatus().lastError.empty());
    EXPECT_EQ(v.chunkStoreView().groupCount(chunkOf(0, kB.x(), kB.z())), 0u);
    // A record with deltas of the newest commit corrupt while the backup's is intact: the open uses the backup
    v.editTerrain(0, addBrush(1152 + 256, 1152));
    ASSERT_EQ(v.commit(), 3u);
    const ChunkEntry* d = v.chunkStoreView().entry(chunkOf(0, 1152 + 256, 1152));
    ASSERT_TRUE(d && d->record);
    const RecordLocation dl = *d->record;
    v.closeSaveDirectory();
    flipByte(*fs, dl.file.path(), size_t(dl.offset) + kRecordHeaderSize + 4);
    {
        World u(unloadTestSetup());
        const SaveOpenResult o = u.openSaveDirectory("mem", memOptions(fs));
        ASSERT_TRUE(o.ok) << o.error;
        EXPECT_TRUE(o.usedBackup);
        EXPECT_EQ(o.commit, 2u);
    }
}

namespace
{
// Crate v2 with a migration (the field extra = 2 * items)
struct CrateV2
{
    static constexpr const char* name = "Crate";
    static constexpr ComponentId id   = ComponentId(UnloadTestComponent::Crate);
    static constexpr uint32_t version = 2;
    int32_t items = 0, extra = 0;
    template <class V>
    void visit(V& v)
    {
        v("items", items);
        v("extra", extra);
    }
};
void registerV2(Registry& r)
{
    r.registerComponent<GrowerState>(Replicate::Host);
    r.registerComponent<CrateV2>(Replicate::Never);
}
void migrateV2(MigrationList& m)
{
    m.add<CrateV2>(1, [](FieldMap& f) { f.set("extra", json11::Json(f.get("items").number_value() * 2)); });
}
void dropCrates(MigrationList& m)
{
    m.addRemoved("Crate", RemovedPolicy::DropEntity);
}
void registerWithoutCrate(Registry& r)
{
    r.registerComponent<GrowerState>(Replicate::Host);
}
}  // namespace

TEST(SaveDirectory, SchemaTablesAndLazyMigration)
{
    auto fs = std::make_shared<MemorySaveFileSystem>();
    PersistentId crateId, plainId;
    {
        auto wp            = makeWorld(unloadTestSetup());
        World& w           = *wp;
        const Entity p     = addPlayer(w, kB);
        const Entity c     = w.spawn<Grower>(kB + Vec3(10, 0, 0));
        const Entity child = w.create();
        w.add<Parent>(child, Parent{c});
        w.add<Crate>(c, Crate{5});
        crateId = w.makePersistent(c);
        plainId = w.makePersistent(w.spawn<Grower>(kB + Vec3(20, 0, 0)));
        w.step();
        teleport(w, p, kA);
        w.step();
        ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
    }
    // A build with Crate v2 and its migration opens the directory: the next commit appends table 1 to the header
    {
        WorldSetup s = unloadTestSetup();
        s.components = registerV2;
        s.migrations = migrateV2;
        auto wp      = makeWorld(s);
        World& w     = *wp;
        ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
        const ChunkEntry* e = w.chunkStoreView().entry(chunkOf(0, kB.x(), kB.z()));
        ASSERT_TRUE(e && e->groups.size() == 2);
        EXPECT_EQ(e->groups[0].schema, 0u);
        w.editTerrain(0, addBrush(1152, 1152));
        ASSERT_EQ(w.commit(), 2u);
        // The group keeps number 0 until it reloads with the migrated value; its rewritten record uses table 1
        Entity player;
        w.registry().view<const Player>().each([&](Entity x, const Player&) { player = x; });
        teleport(w, player, kB);
        w.step();
        const Entity c = w.resolve(crateId);
        ASSERT_TRUE(c.valid());
        EXPECT_EQ(w.read<CrateV2>(c).extra, 10);
        teleport(w, player, kA);
        w.step();
        ASSERT_EQ(w.commit(), 3u);
        EXPECT_EQ(w.chunkStoreView().tables().size(), 2u);
        e = w.chunkStoreView().entry(chunkOf(0, kB.x(), kB.z()));
        ASSERT_TRUE(e && !e->groups.empty());
        EXPECT_EQ(e->groups[0].schema, 1u);
        EXPECT_EQ(e->summary.groups[0].schema, 1u);
    }
    // A build without Crate (Removed: drop the entity) drops that entity and its descendants on reload
    {
        WorldSetup s = unloadTestSetup();
        s.components = registerWithoutCrate;
        s.migrations = dropCrates;
        auto wp      = makeWorld(s);
        World& w     = *wp;
        ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
        Entity player;
        w.registry().view<const Player>().each([&](Entity x, const Player&) { player = x; });
        teleport(w, player, kB);
        w.step();
        EXPECT_FALSE(w.resolve(crateId).valid());
        EXPECT_TRUE(w.resolve(plainId).valid());
    }
}

TEST(SaveDirectory, Compaction)
{
    auto wp        = makeWorld(unloadTestSetup());
    World& w       = *wp;
    const Entity p = addPlayer(w, kB);
    // One large group (random scales and values): every commit that rewrites its chunk appends ~130 KB
    const Entity root = w.spawn<Grower>(kB + Vec3(10, 0, 0));
    Rng rng(3);
    for (int i = 0; i < 10000; ++i)
    {
        const Entity c = w.create();
        w.add<Parent>(c, Parent{root});
        w.get<Transform>(c).setScale(Vec3(rng.uniform(), rng.uniform(), rng.uniform()));
        GrowerState g;
        g.grown = rng.uniform();
        w.add<GrowerState>(c, g);
    }
    w.step();
    teleport(w, p, kA);
    w.step();
    auto fs = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
    const RegionKey first = w.chunkStoreView().entry(chunkOf(0, kB.x(), kB.z()))->record->file;
    RegionKey compacted;
    int compactedAt = -1;
    for (int i = 0; i < 30 && compactedAt < 0; ++i)
    {
        w.editTerrain(0, addBrush(kB.x() + 60, 900 + i));
        w.commit();
        if (getenv("S25DBG"))
        {
            uint64_t sz    = 0;
            const auto* en = w.chunkStoreView().entry(chunkOf(0, kB.x(), kB.z()));
            fs->fileSize(en->record->file.path(), sz, nullptr);
            printf("commit %d: %s size %llu record %u groups %zu dirty %d\n", i, en->record->file.path().c_str(),
                   (unsigned long long)sz, en->record->length, en->groups.size(), int(en->dirty));
        }
        if (!(w.chunkStoreView().entry(chunkOf(0, kB.x(), kB.z()))->record->file == first))
        {
            compactedAt = i;
            compacted   = w.chunkStoreView().entry(chunkOf(0, kB.x(), kB.z()))->record->file;
        }
    }
    ASSERT_GE(compactedAt, 0);
    EXPECT_EQ(w.chunkStoreView().stats().compactions, 1u);
    EXPECT_EQ(compacted.generation, first.generation + 1);
    // The old generation survives one more commit (the backup names it), then it is removed
    EXPECT_TRUE(hasFile(*fs, first.path()));
    w.editTerrain(0, addBrush(kB.x() + 70, 920));
    w.commit();
    EXPECT_FALSE(hasFile(*fs, first.path()));
    uint64_t size = 0;
    ASSERT_TRUE(fs->fileSize(compacted.path(), size, nullptr));
    EXPECT_LT(size, kCompactMinBytes);
    const std::vector<uint8_t> blob = w.save();
    w.closeSaveDirectory();
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    ASSERT_TRUE(v.openSaveDirectory("mem", memOptions(fs)).ok);
    EXPECT_EQ(v.save(), blob);
}

TEST(SaveDirectory, AsyncEqualsSynchronous)
{
    auto run = [](bool synchronous)
    {
        Scene sc = makeScene();
        World& w = *sc.w;
        auto fs  = std::make_shared<MemorySaveFileSystem>();
        EXPECT_TRUE(w.openSaveDirectory("mem", memOptions(fs, synchronous)).ok);
        w.waitForCommits();
        teleport(w, sc.player, kB);
        w.step();
        w.editTerrain(0, addBrush(kB.x() + 60, kB.z() + 60));
        w.commit();
        w.waitForCommits();
        teleport(w, sc.player, kA);
        w.step();
        w.commit();
        w.waitForCommits();
        w.closeSaveDirectory();
        std::map<std::string, std::vector<uint8_t>> files;
        for (const std::string& f : fs->files()) files[f] = fs->fileBytes(f);
        return files;
    };
    EXPECT_EQ(run(false), run(true));
}

TEST(SaveDirectory, CommitWhileInFlight)
{
    Scene sc = makeScene();
    World& w = *sc.w;
    auto fs  = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs, false)).ok);
    ASSERT_TRUE(w.waitForCommits());
    w.saveDirectory()->ioQueue().hookForTests = [] { latch().wait(); };
    latch().close();
    w.editTerrain(0, addBrush(1152 + 256, 1152));
    ASSERT_EQ(w.commit(), 2u);
    // A second commit is deferred; a chunk edited and a root unloaded while the commit is in flight
    EXPECT_EQ(w.commit(), 0u);
    w.editTerrain(0, addBrush(1152 + 512, 1152));
    const PersistentId moved = w.makePersistent(w.spawn<Grower>(kA + Vec3(20, 0, 0)));
    teleport(w, sc.player, kA + Vec3(0, 0, 3000));
    // The groups at B (disk-only since commit 1) reload while commit 2 is in flight... the player goes there below
    latch().waitEntered(1);
    latch().open();
    w.step();
    ASSERT_TRUE(w.waitForCommits());
    w.saveDirectory()->ioQueue().hookForTests = nullptr;
    // The deferred commit runs at the first step boundary after the completion
    w.step();
    ASSERT_TRUE(w.waitForCommits());
    EXPECT_GE(w.saveStatus().committed, 3u);
    EXPECT_FALSE(w.resolve(moved).valid());
    // The groups at B reload while commit c + 1 is in flight: afterwards the entry has no groups and the entities exist
    // once
    latch().close();
    w.saveDirectory()->ioQueue().hookForTests = [] { latch().wait(); };
    w.editTerrain(0, addBrush(1152 + 768, 1152));
    const uint64_t c = w.commit();
    ASSERT_NE(c, 0u);
    teleport(w, sc.player, kB);
    w.step();
    for (PersistentId id : sc.stored) EXPECT_TRUE(w.resolve(id).valid());
    latch().waitEntered(1);
    latch().open();
    ASSERT_TRUE(w.waitForCommits());
    w.saveDirectory()->ioQueue().hookForTests = nullptr;
    const uint64_t next                       = w.commit();
    ASSERT_TRUE(w.waitForCommits());
    EXPECT_EQ(w.chunkStoreView().groupCount(chunkOf(0, kB.x(), kB.z())), 0u);
    const std::vector<uint8_t> blob = w.save();
    w.closeSaveDirectory();
    World v(unloadTestSetup());
    v.setStreamingSynchronous(true);
    const SaveOpenResult o = v.openSaveDirectory("mem", memOptions(fs));
    ASSERT_TRUE(o.ok) << o.error;
    EXPECT_EQ(o.commit, next);
    EXPECT_EQ(v.save(), blob);
    for (PersistentId id : sc.stored) EXPECT_TRUE(v.resolve(id).valid());
}

TEST(SaveDirectory, FailedCommitRetries)
{
    // The operations of one commit counted once, then a commit failed at each of them in turn
    uint64_t ops = 0;
    {
        Scene sc = makeScene();
        auto fs  = std::make_shared<MemorySaveFileSystem>();
        ASSERT_TRUE(sc.w->openSaveDirectory("mem", memOptions(fs)).ok);
        sc.w->editTerrain(0, addBrush(kB.x() + 60, kB.z() + 60));
        const uint64_t before = fs->operations();
        ASSERT_EQ(sc.w->commit(), 2u);
        ops = fs->operations() - before;
    }
    ASSERT_GT(ops, 4u);
    for (uint64_t n = 0; n < ops; ++n)
        for (bool torn : {false, true})
        {
            Scene sc = makeScene();
            World& w = *sc.w;
            auto fs  = std::make_shared<MemorySaveFileSystem>();
            ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
            w.editTerrain(0, addBrush(kB.x() + 60, kB.z() + 60));
            fs->failAfter(int64_t(n), torn);
            const uint64_t c = w.commit();
            fs->failAfter(-1);
            const SaveStatus s = w.saveStatus();
            ASSERT_EQ(c, 2u);
            if (s.committed == 1)
            {
                // Before the commit point: failed, the dirty chunks stay dirty, the world continues
                EXPECT_FALSE(s.lastError.empty()) << n;
                EXPECT_GT(s.dirtyChunks, 0u) << n;
                ASSERT_EQ(w.commit(), 2u) << n;
            }
            // After the commit point (the directory sync, a remove): done, the error reported
            EXPECT_EQ(w.saveStatus().committed, 2u) << n;
            const std::vector<uint8_t> blob = w.save();
            w.closeSaveDirectory();
            World v(unloadTestSetup());
            v.setStreamingSynchronous(true);
            const SaveOpenResult o = v.openSaveDirectory("mem", memOptions(fs));
            ASSERT_TRUE(o.ok) << n << ": " << o.error;
            EXPECT_EQ(o.commit, 2u) << n;
            EXPECT_EQ(v.save(), blob) << n;
        }
}

TEST(SaveDirectory, EveryGeneratedChunkHasRecord)
{
    using namespace StreamingTest;
    auto fs = std::make_shared<MemorySaveFileSystem>();
    ChunkId visited;
    {
        // test_terrain v1 has another generator identity: its own cache (once per process)
        static std::shared_ptr<ContentCache> v1Cache = std::make_shared<ContentCache>();
        WorldSetup s                                 = streamingTestSetup(v1Cache, 1);
        auto wp                                      = makeWorld(s);
        World& w                                     = *wp;
        StreamingTest::addPlayer(w, Vec3(5150, 0, 1400));
        w.step();
        visited = StreamingTest::chunkOf(0, 5150, 1400);
        ASSERT_TRUE(w.residency().ready(visited));
        ASSERT_NE(w.chunkRecord(visited), nullptr);
        EXPECT_EQ(w.chunkRecord(visited)->placementVersion, 1);
        EXPECT_GT(w.chunkStoreView().stats().autoRecords, 0u);
        ASSERT_TRUE(w.openSaveDirectory("mem", memOptions(fs)).ok);
    }
    WorldSetup s           = streamingTestSetup(sharedStreamingCache(), 2);
    auto wp                = makeWorld(s);
    World& w               = *wp;
    const SaveOpenResult r = w.openSaveDirectory("mem", memOptions(fs));
    ASSERT_TRUE(r.ok) << r.error;
    // Summary-only entries: no record read for them (only the space records of the unloaded spaces)
    EXPECT_EQ(r.recordsRead, w.chunkStoreView().unloadedSpaces().size());
    EXPECT_EQ(w.effectivePlacementVersion(visited), 1);
    EXPECT_EQ(w.effectivePlacementVersion(StreamingTest::chunkOf(0, 9150, 1400)), 2);
}

TEST(SaveDirectory, LockedAndRefusals)
{
    Scene sc = makeScene();
    auto fs  = std::make_shared<MemorySaveFileSystem>();
    ASSERT_TRUE(sc.w->openSaveDirectory("mem", memOptions(fs)).ok);
    World v(unloadTestSetup());
    const SaveOpenResult r = v.openSaveDirectory("mem", memOptions(fs));
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("memory"), std::string::npos);
    EXPECT_FALSE(v.hasSaveDirectory());
    // A second open of the same world is refused
    EXPECT_FALSE(sc.w->openSaveDirectory("mem", memOptions(fs)).ok);
    sc.w->closeSaveDirectory();
    EXPECT_TRUE(v.openSaveDirectory("mem", memOptions(fs)).ok);
    World u(unloadTestSetup());
    EXPECT_EQ(u.commit(), 0u);
}

TEST(SaveDirectory, CommittedDirectoryOpens)
{
    // data/tests/engine/save_dir_s25/ (made by this test with SAIGA_UPDATE_GOLDEN=1, only for a new format version): a
    // stored group, edited chunks, an unloaded space; copied into a memory file system (the source tree stays
    // untouched)
    const std::string dir = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine/save_dir_s25";
    auto build            = [](World& w, Entity& player, PersistentId& id)
    {
        player                                  = addPlayer(w, kB);
        id                                      = w.makePersistent(w.spawn<Grower>(kB + Vec3(10, 0, 0)));
        w.get<GrowerState>(w.resolve(id)).grown = 2.5;
        w.step();
        teleport(w, player, kA);
        w.editTerrain(0, addBrush(1152, 1152));
        w.step(2);
    };
    WorldSetup s          = unloadTestSetup();
    s.environment.enabled = true;
    if (getenv("SAIGA_UPDATE_GOLDEN"))
    {
        std::filesystem::remove_all(string_to_path(dir));
        auto wp = makeWorld(s);
        Entity p;
        PersistentId id;
        build(*wp, p, id);
        SaveDirectoryOptions o = memOptions(nullptr);
        o.fileSystem           = nullptr;
        ASSERT_TRUE(wp->openSaveDirectory(dir, o).ok);
        wp->closeSaveDirectory();
        std::filesystem::remove(string_to_path(dir + "/lock"));
        GTEST_SKIP() << "wrote " << dir;
    }
    auto fs = std::make_shared<MemorySaveFileSystem>();
    for (const auto& entry : std::filesystem::recursive_directory_iterator(string_to_path(dir)))
    {
        if (!entry.is_regular_file()) continue;
        std::ifstream in(entry.path(), std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        fs->putFile(path_to_string(std::filesystem::relative(entry.path(), string_to_path(dir))), bytes);
    }
    ASSERT_TRUE(fs->exists("world.1.sgw"));
    auto ref = makeWorld(s);
    Entity p;
    PersistentId id;
    build(*ref, p, id);
    auto wp                = makeWorld(s);
    const SaveOpenResult r = wp->openSaveDirectory("mem", memOptions(fs));
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_EQ(wp->save(), ref->save());
    EXPECT_EQ(wp->terrainSample(0, 576, 576), ref->terrainSample(0, 576, 576));
    EXPECT_FALSE(wp->spaceLoaded(1));
    teleport(*wp, p, kB);
    teleport(*ref, p, kB);
    wp->step();
    ref->step();
    ASSERT_TRUE(wp->resolve(id).valid());
    EXPECT_GT(wp->read<GrowerState>(wp->resolve(id)).grown, 2.5);
    EXPECT_EQ(wp->save(), ref->save());
}
