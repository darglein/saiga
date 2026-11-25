/**
 * Copyright (c) 2026 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/string_to_path.h"
#include "saiga/core/util/zlib/zip_archive.h"
#include "saiga/core/util/zlib/zip_stream_writer.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace Saiga
{
namespace
{
struct TempFile
{
    explicit TempFile(const std::string& name) : path(std::filesystem::temp_directory_path() / ("saiga_" + name)) {}
    ~TempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    std::filesystem::path path;
};

std::vector<uint8_t> file_bytes(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<uint8_t> pattern(size_t size, uint8_t seed)
{
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint8_t>(seed + i * 31 + (i >> 7));
    return data;
}

// `name` is UTF-8, as the writer stores it. Converted with string_to_path, since a plain std::string -> path
// conversion uses the ANSI code page on Windows.
std::vector<uint8_t> read_entry(const ZipArchive& archive, const std::string& name)
{
    auto [found, file] = archive.find_file(string_to_path(name));
    if (!found) return {};
    std::vector<uint8_t> data(file.uncompressed_size);
    if (!data.empty() && !file.read_all(data.data())) return {};
    return data;
}

uint16_t get16(const std::vector<uint8_t>& b, size_t at)
{
    return static_cast<uint16_t>(b[at] | (b[at + 1] << 8));
}

uint32_t get32(const std::vector<uint8_t>& b, size_t at)
{
    return get16(b, at) | (static_cast<uint32_t>(get16(b, at + 2)) << 16);
}

struct VectorSink : ZipStreamSink
{
    void write(const void* data, size_t size) override
    {
        const auto* p = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + size);
    }
    std::vector<uint8_t> bytes;
};

// Skips runs of zeros instead of writing them, which leaves holes in a sparse file.
struct SparseFileSink : ZipStreamSink
{
    explicit SparseFileSink(const std::filesystem::path& path) : out(path, std::ios::binary | std::ios::trunc) {}
    void write(const void* data, size_t size) override
    {
        const auto* p = static_cast<const uint8_t*>(data);
        if (std::all_of(p, p + size, [](uint8_t v) { return v == 0; }))
        {
            out.seekp(static_cast<std::streamoff>(size), std::ios::cur);
        }
        else
        {
            out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        }
        if (!out) throw std::runtime_error("write failed");
    }
    std::ofstream out;
};

// Fails on call `fail_at` and after.
struct FailingSink : ZipStreamSink
{
    explicit FailingSink(int fail_at) : fail_at(fail_at) {}
    void write(const void*, size_t) override
    {
        if (++calls >= fail_at) throw std::runtime_error("disk full");
    }
    int fail_at;
    int calls = 0;
};
}  // namespace

TEST(ZipStreamWriter, EmptyArchiveHasOnlyTheEndRecord)
{
    TempFile tmp("zip_stream_empty.zip");
    {
        ZipStreamWriter zip(tmp.path);
        zip.finish();
        EXPECT_EQ(zip.offset(), 22u);
    }
    ZipArchive archive(tmp.path, ZipMode::Read);
    ASSERT_TRUE(archive.is_open());
    EXPECT_EQ(archive.file_count(), 0);
}

TEST(ZipStreamWriter, WholeAndStreamedEntriesReadBack)
{
    TempFile tmp("zip_stream_entries.zip");
    const auto a      = pattern(1000, 1);
    const auto b      = pattern(300000, 2);
    uint64_t a_offset = 0, b_offset = 0;
    {
        ZipStreamWriter zip(tmp.path);
        a_offset = zip.add_entry("a.bin", a.data(), a.size());
        b_offset = zip.begin_entry("dir/b.bin");
        for (size_t at = 0; at < b.size(); at += 4096)
        {
            zip.write(b.data() + at, std::min<size_t>(4096, b.size() - at));
            // An empty write must not reset the running CRC.
            zip.write(nullptr, 0);
        }
        zip.end_entry();
        zip.add_entry("empty", nullptr, 0);
        zip.begin_entry("streamed_empty");
        zip.end_entry();
        zip.add_entry(
            "\xC3\xBC"
            "n\xC3\xAF"
            "code.txt",
            "x", 1);
        zip.finish();
    }

    // The returned offsets point at the entries' bytes in the file.
    const auto bytes = file_bytes(tmp.path);
    ASSERT_LE(b_offset + b.size(), bytes.size());
    EXPECT_TRUE(std::equal(a.begin(), a.end(), bytes.begin() + a_offset));
    EXPECT_TRUE(std::equal(b.begin(), b.end(), bytes.begin() + b_offset));

    ZipArchive archive(tmp.path, ZipMode::Read);
    ASSERT_TRUE(archive.is_open());
    EXPECT_EQ(archive.file_count(), 5);
    EXPECT_EQ(read_entry(archive, "a.bin"), a);
    EXPECT_EQ(read_entry(archive, "dir/b.bin"), b);
    EXPECT_TRUE(archive.find_file("empty").first);
    EXPECT_TRUE(archive.find_file("streamed_empty").first);
    EXPECT_EQ(read_entry(archive,
                         "\xC3\xBC"
                         "n\xC3\xAF"
                         "code.txt"),
              std::vector<uint8_t>{'x'});
}

TEST(ZipStreamWriter, ArchiveNestedInAStreamedEntry)
{
    TempFile tmp("zip_stream_nested.zip");
    const auto inner_data = pattern(50000, 3);
    {
        ZipStreamWriter outer(tmp.path);
        outer.add_entry("header.json", "{}", 2);
        outer.begin_entry("inner.zip");
        ZipStreamWriter inner(&outer);
        inner.add_entry("x/data.bin", inner_data.data(), inner_data.size());
        inner.finish();
        outer.end_entry();
        outer.finish();
    }

    ZipArchive outer(tmp.path, ZipMode::Read);
    ASSERT_TRUE(outer.is_open());
    EXPECT_EQ(read_entry(outer, "header.json"), (std::vector<uint8_t>{'{', '}'}));
    auto [found, inner_file] = outer.find_file("inner.zip");
    ASSERT_TRUE(found);
    ZipArchive inner(inner_file);
    ASSERT_TRUE(inner.is_open());
    EXPECT_EQ(read_entry(inner, "x/data.bin"), inner_data);
}

TEST(ZipStreamWriter, ManyEntriesUseTheZip64EndRecord)
{
    TempFile tmp("zip_stream_many.zip");
    constexpr int kCount = 70000;
    {
        ZipStreamWriter zip(tmp.path);
        for (int i = 0; i < kCount; ++i)
        {
            const std::string body = std::to_string(i);
            zip.add_entry("e/" + body, body.data(), body.size());
        }
        zip.finish();
    }

    const auto bytes = file_bytes(tmp.path);
    ASSERT_GE(bytes.size(), 22u + 20u);
    const size_t end = bytes.size() - 22;
    EXPECT_EQ(get32(bytes, end), 0x06054b50u);
    EXPECT_EQ(get16(bytes, end + 10), 0xFFFFu);
    EXPECT_EQ(get32(bytes, end - 20), 0x07064b50u);

    ZipArchive archive(tmp.path, ZipMode::Read);
    ASSERT_TRUE(archive.is_open());
    EXPECT_EQ(archive.file_count(), kCount);
    EXPECT_EQ(read_entry(archive, "e/69999"), (std::vector<uint8_t>{'6', '9', '9', '9', '9'}));
}

// Writes zeros as holes, so the 4 GiB entry costs no disk space where the file system supports sparse files.
TEST(ZipStreamWriter, EntryOver4GiBUsesZip64Fields)
{
#ifdef _WIN32
    GTEST_SKIP() << "needs a sparse file";
#endif
    TempFile tmp("zip_stream_large.zip");
    constexpr uint64_t kLarge = (uint64_t(1) << 32) + 12345;
    const auto tail           = pattern(777, 4);
    {
        SparseFileSink sink(tmp.path);
        ASSERT_TRUE(sink.out.is_open());
        ZipStreamWriter zip(&sink);
        const std::vector<uint8_t> zeros(1 << 20, 0);
        zip.begin_entry("large.bin");
        for (uint64_t left = kLarge; left > 0;)
        {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(left, zeros.size()));
            zip.write(zeros.data(), n);
            left -= n;
        }
        zip.end_entry();
        // Lies past 4 GiB, so its central header needs a 64-bit offset.
        zip.add_entry("tail.bin", tail.data(), tail.size());
        zip.finish();
    }

    ZipArchive archive(tmp.path, ZipMode::Read);
    ASSERT_TRUE(archive.is_open());
    auto [found, large] = archive.find_file("large.bin");
    ASSERT_TRUE(found);
    EXPECT_EQ(large.uncompressed_size, kLarge);
    EXPECT_EQ(read_entry(archive, "tail.bin"), tail);
}

TEST(ZipStreamWriter, RejectsCallsOutOfOrder)
{
    VectorSink sink;
    ZipStreamWriter zip(&sink);
    EXPECT_THROW(zip.write("a", 1), std::logic_error);
    EXPECT_THROW(zip.end_entry(), std::logic_error);
    EXPECT_THROW(zip.add_entry("", "a", 1), std::logic_error);
    zip.begin_entry("a");
    EXPECT_THROW(zip.begin_entry("b"), std::logic_error);
    EXPECT_THROW(zip.add_entry("b", "b", 1), std::logic_error);
    EXPECT_THROW(zip.finish(), std::logic_error);
    zip.end_entry();
    zip.finish();
    EXPECT_EQ(zip.offset(), sink.bytes.size());
    EXPECT_THROW(zip.add_entry("c", "c", 1), std::logic_error);
}

TEST(ZipStreamWriter, RefusesFurtherCallsAfterASinkError)
{
    FailingSink sink(2);
    ZipStreamWriter zip(&sink);
    zip.begin_entry("a");  // first sink call: the local header
    EXPECT_THROW(zip.write("x", 1), std::runtime_error);
    EXPECT_THROW(zip.write("x", 1), std::runtime_error);
    EXPECT_THROW(zip.finish(), std::runtime_error);
    EXPECT_EQ(sink.calls, 2);  // a failed writer does not call the sink again
}

TEST(ZipStreamWriter, RejectsANullSink)
{
    EXPECT_THROW(ZipStreamWriter(static_cast<ZipStreamSink*>(nullptr)), std::logic_error);
}

}  // namespace Saiga
