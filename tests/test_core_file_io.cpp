/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/zlib/unzip_wrapper.h"
#include "saiga/core/util/zlib/zip_archive.h"

#include "gtest/gtest.h"

#include <cstring>
#include <fstream>

using namespace Saiga;

static std::filesystem::path TestDir()
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_core_file_io";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

static void WriteText(const std::filesystem::path& file, const std::string& content)
{
    std::ofstream strm(file, std::ios::binary);
    strm << content;
}

TEST(FileIO, LoadFileStringArray)
{
    auto dir = TestDir();
    WriteText(dir / "a.txt", "a\nb\n");
    WriteText(dir / "b.txt", "a\nb");
    WriteText(dir / "empty.txt", "");

    EXPECT_EQ(File::loadFileStringArray(dir / "a.txt"), (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(File::loadFileStringArray(dir / "b.txt"), (std::vector<std::string>{"a", "b"}));
    EXPECT_TRUE(File::loadFileStringArray(dir / "empty.txt").empty());
    EXPECT_TRUE(File::loadFileStringArray(dir / "missing.txt").empty());
}

TEST(FileIO, LoadFileString)
{
    auto dir = TestDir();
    WriteText(dir / "a.txt", "a\nb\n");
    WriteText(dir / "bom.txt", "\xEF\xBB\xBFx");
    EXPECT_EQ(File::loadFileString(dir / "a.txt"), "a\nb\n");
    EXPECT_EQ(File::loadFileString(dir / "bom.txt"), "x\n");
}

TEST(FileIO, DirectoryInsteadOfFile)
{
    // Used to loop forever (string functions) or throw std::length_error (binary).
    auto dir = TestDir();
    EXPECT_TRUE(File::loadFileStringArray(dir).empty());
    EXPECT_TRUE(File::loadFileString(dir).empty());
    EXPECT_TRUE(File::loadFileBinary(dir).empty());
}

TEST(FileIO, BinaryRoundtrip)
{
    auto dir = TestDir();
    std::vector<char> data(1000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = char(i * 7);
    File::saveFileBinary(dir / "data.bin", data.data(), data.size());
    EXPECT_EQ(File::loadFileBinary(dir / "data.bin"), data);
}

TEST(FileIO, DirectoryGetFilesWithSymlink)
{
    auto dir = TestDir();
    std::filesystem::create_directories(dir / "sub" / "other");
    WriteText(dir / "sub" / "other" / "target.tif", "x");
    WriteText(dir / "sub" / "plain.tif", "x");
    std::error_code ec;
    std::filesystem::create_symlink(dir / "sub" / "other" / "target.tif", dir / "sub" / "link.tif", ec);
    if (ec) GTEST_SKIP() << "symlinks not supported: " << ec.message();

    Directory d(dir / "sub");
    auto files = d.getFiles();
    std::sort(files.begin(), files.end());
    EXPECT_EQ(files, (std::vector<std::filesystem::path>{"link.tif", "plain.tif"}));
    EXPECT_EQ(d.getDirectories(), (std::vector<std::filesystem::path>{"other"}));
}

#if defined(SAIGA_USE_LIBZIP)

// Overwrites the uncompressed size of the only entry in both the local file header and the central directory.
static void PatchUncompressedSize(const std::filesystem::path& file, uint32_t new_size)
{
    auto data   = File::loadFileBinary(file);
    int patched = 0;
    for (size_t i = 0; i + 4 <= data.size(); ++i)
    {
        uint32_t sig;
        std::memcpy(&sig, &data[i], 4);
        if (sig == 0x04034b50 && i + 26 <= data.size())
        {
            std::memcpy(&data[i + 22], &new_size, 4);
            patched++;
        }
        if (sig == 0x02014b50 && i + 28 <= data.size())
        {
            std::memcpy(&data[i + 24], &new_size, 4);
            patched++;
        }
    }
    ASSERT_EQ(patched, 2);
    File::saveFileBinary(file, data.data(), data.size());
}

static std::filesystem::path WriteStoredZip(const std::filesystem::path& dir, const std::vector<char>& data)
{
    auto path = dir / "test.zip";
    ZipArchive archive(path, ZipMode::Write);
    EXPECT_TRUE(archive.is_open());
    EXPECT_TRUE(archive.add_file("entry.bin", (void*)data.data(), data.size(), ZipCompressionMethod::UNCOMPRESSED));
    EXPECT_TRUE(archive.close());
    return path;
}

TEST(FileIO, ZipReadAll)
{
    auto dir = TestDir();
    std::vector<char> data(5000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = char(i * 13);
    auto path = WriteStoredZip(dir, data);

    ZipArchive archive(path, ZipMode::Read);
    auto [found, file] = archive.find_file("entry.bin");
    ASSERT_TRUE(found);
    ASSERT_EQ(file.uncompressed_size, data.size());
    std::vector<char> out(file.uncompressed_size);
    EXPECT_TRUE(file.read_all(out.data()));
    EXPECT_EQ(out, data);

    // open + read in pieces
    ASSERT_TRUE(file.open());
    std::vector<char> part(1000);
    EXPECT_TRUE(file.read(part.data(), part.size()));
    EXPECT_TRUE(std::equal(part.begin(), part.end(), data.begin()));
    file.close();
}

TEST(FileIO, ZipReadAllDoesNotWritePastDeclaredSize)
{
    auto dir = TestDir();
    std::vector<char> data(5000, 'x');
    auto path = WriteStoredZip(dir, data);
    // The archive now claims that the entry has 100 bytes, but it contains 5000.
    PatchUncompressedSize(path, 100);

    ZipArchive archive(path, ZipMode::Read);
    auto files = archive.get_files();
    ASSERT_EQ(files.size(), 1);
    ASSERT_EQ(files[0].uncompressed_size, 100);

    // A buffer of the declared size followed by a guard region.
    std::vector<char> buffer(100 + 10000, 'g');
    EXPECT_FALSE(files[0].read_all(buffer.data()));
    for (size_t i = 100; i < buffer.size(); ++i)
    {
        ASSERT_EQ(buffer[i], 'g') << "read_all wrote past the declared size at offset " << i;
    }
}

TEST(FileIO, ZipTruncatedEntry)
{
    auto dir = TestDir();
    std::vector<char> data(500, 'y');
    auto path = WriteStoredZip(dir, data);
    // The archive claims more data than it contains.
    PatchUncompressedSize(path, 1000);

    ZipArchive archive(path, ZipMode::Read);
    auto files = archive.get_files();
    ASSERT_EQ(files.size(), 1);

    std::vector<char> buffer(1000);
    EXPECT_FALSE(files[0].read_all(buffer.data()));

    // Reading more than available returns false instead of aborting.
    if (files[0].open())
    {
        EXPECT_FALSE(files[0].read(buffer.data(), buffer.size()));
        files[0].close();
    }
}

TEST(FileIO, ZipCrcMismatch)
{
    auto dir = TestDir();
    std::vector<char> data(500, 'z');
    auto path = WriteStoredZip(dir, data);

    // Flip one byte of the stored data, which follows the local header (30 bytes + name length).
    auto bytes = File::loadFileBinary(path);
    bytes[30 + std::string("entry.bin").size() + 10] ^= 1;
    File::saveFileBinary(path, bytes.data(), bytes.size());

    ZipArchive archive(path, ZipMode::Read);
    auto files = archive.get_files();
    ASSERT_EQ(files.size(), 1);
    std::vector<char> buffer(500);
    EXPECT_FALSE(files[0].read_all(buffer.data()));
}

TEST(FileIO, ZipOpenFailureSetsError)
{
    auto dir = TestDir();
    ZipArchive archive(dir / "does_not_exist.zip", ZipMode::Read);
    EXPECT_FALSE(archive.is_open());
    EXPECT_FALSE(archive.get_last_error().empty());
}

TEST(FileIO, ZipAddFolder)
{
    auto dir = TestDir();
    std::filesystem::create_directories(dir / "folder" / "nested");
    WriteText(dir / "folder" / "a.txt", "a");
    WriteText(dir / "folder" / "nested" / "b.txt", "bb");
    {
        ZipArchive archive(dir / "folder.zip", ZipMode::Write);
        EXPECT_TRUE(archive.add_folder("root", dir / "folder", ZipCompressionMethod::ZLIB));
        EXPECT_TRUE(archive.close());
    }
    ZipArchive archive(dir / "folder.zip", ZipMode::Read);
    EXPECT_TRUE(archive.find_file("root/a.txt").first);
    EXPECT_TRUE(archive.find_file("root/nested/b.txt").first);
    EXPECT_EQ(archive.file_count(), 2);

    ZipArchive missing(dir / "missing.zip", ZipMode::Write);
    EXPECT_FALSE(missing.add_folder("", dir / "does_not_exist", ZipCompressionMethod::ZLIB));
}

TEST(FileIO, UnzipToMemory)
{
    auto dir = TestDir();
    std::vector<char> data(25000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = char(i * 3);
    auto path = WriteStoredZip(dir, data);

    auto info = UnzipInfo(path.string());
    ASSERT_EQ(info.size(), 1);
    EXPECT_EQ(info[0].name, "entry.bin");
    EXPECT_EQ(info[0].uncompressed_size, data.size());

    auto files = UnzipToMemory(path.string());
    ASSERT_EQ(files.size(), 1);
    EXPECT_EQ(files[0].data, data);

    Unzipfile single = info[0];
    UnzipSingleFileToMemory(path.string(), single);
    EXPECT_EQ(single.data, data);

    std::vector<char> user_buffer(data.size());
    single.user_data_ptr = user_buffer.data();
    single.data.clear();
    UnzipSingleFileToMemory(path.string(), single);
    EXPECT_EQ(user_buffer, data);
    EXPECT_TRUE(single.data.empty());
}

#    if defined(__linux__)
static int OpenFileDescriptors()
{
    std::error_code ec;
    int count = 0;
    for (auto it = std::filesystem::directory_iterator("/proc/self/fd", ec); !ec && it != decltype(it)(); ++it) count++;
    return count;
}

TEST(FileIO, UnzipToMemoryClosesFile)
{
    // UnzipToMemory() never closed the zip file after extracting all entries.
    auto dir  = TestDir();
    auto path = WriteStoredZip(dir, std::vector<char>(100, 'u'));
    auto info = UnzipInfo(path.string());
    ASSERT_EQ(info.size(), 1);

    int before = OpenFileDescriptors();
    for (int i = 0; i < 10; ++i) UnzipToMemory(path.string(), info);
    EXPECT_EQ(OpenFileDescriptors(), before);
}
#    endif
#endif
