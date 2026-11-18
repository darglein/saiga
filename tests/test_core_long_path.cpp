/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/image/templatedImage.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/long_path.h"
#include "saiga/core/util/zlib/zip_archive.h"

#include "gtest/gtest.h"

#include <fstream>
#include <random>

// Built twice on Windows: test_core_long_path is not long path aware, so it behaves like a system without
// LongPathsEnabled, and test_core_long_path_aware has the manifest of saiga_enable_long_paths. Both must pass.

using namespace Saiga;

#if defined(_WIN32)
static const std::wstring verbatim = LR"(\\?\)";
#endif

// A short folder of this process in the temp folder, removed at exit: ctest runs both executables at the same time.
struct ProcessDir
{
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                ("saiga_test_core_long_path_" + std::to_string(std::random_device()()));
    ProcessDir() { std::filesystem::create_directories(dir); }
    ~ProcessDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(os_path(dir, LongPathPrefix::Always), ec);
    }
};

static const std::filesystem::path& TestDir()
{
    static const ProcessDir process_dir;
    return process_dir.dir;
}

// A folder below TestDir() whose path is longer than 300 characters.
static std::filesystem::path LongDir()
{
    auto base = TestDir() / "long";
    std::error_code ec;
    std::filesystem::remove_all(os_path(base, LongPathPrefix::Always), ec);

    auto dir = base;
    for (char c = 'a'; c < 'a' + 8; ++c)
    {
        dir /= std::string(40, c);
    }
    EXPECT_GT(dir.native().size(), 300u);
    EXPECT_TRUE(std::filesystem::create_directories(os_path(dir), ec)) << ec.message();
    return dir;
}

static void WriteText(const std::filesystem::path& file, const std::string& content)
{
    std::ofstream strm(os_path(file), std::ios::binary);
    strm << content;
}

TEST(LongPath, ShortPathsAreUnchanged)
{
    auto relative = std::filesystem::path("folder") / "file.txt";
    EXPECT_EQ(os_path(relative), relative);

    auto absolute = std::filesystem::temp_directory_path() / "file.txt";
    EXPECT_EQ(os_path(absolute), absolute);
    EXPECT_EQ(os_path(std::filesystem::path()), std::filesystem::path());
}

#if defined(_WIN32)
TEST(LongPath, LongPathsGetThePrefix)
{
    std::wstring long_name(300, L'x');
    auto path = os_path(std::filesystem::path(L"C:/folder/../other/" + long_name));
    EXPECT_EQ(path.native(), verbatim + L"C:\\other\\" + long_name);

    path = os_path(std::filesystem::path(L"\\\\server\\share\\" + long_name));
    EXPECT_EQ(path.native(), verbatim + L"UNC\\server\\share\\" + long_name);

    // Already prefixed paths and device paths are left alone.
    EXPECT_EQ(os_path(path), path);
    std::filesystem::path device = LR"(\\.\pipe\name)";
    EXPECT_EQ(os_path(device), device);
}

TEST(LongPath, AlwaysPrefixesShortPaths)
{
    EXPECT_EQ(os_path(std::filesystem::path(L"C:/a/b.txt"), LongPathPrefix::Always).native(),
              verbatim + L"C:\\a\\b.txt");
    EXPECT_EQ(os_path(std::filesystem::path(L"\\\\server\\share\\b.txt"), LongPathPrefix::Always).native(),
              verbatim + L"UNC\\server\\share\\b.txt");
}

TEST(LongPath, StripOsPrefix)
{
    EXPECT_EQ(strip_os_prefix(verbatim + L"C:\\a\\b.txt").native(), L"C:\\a\\b.txt");
    EXPECT_EQ(strip_os_prefix(verbatim + L"UNC\\server\\share\\b.txt").native(), L"\\\\server\\share\\b.txt");
    // Volume paths have no form without the prefix.
    std::wstring volume = verbatim + L"Volume{01234567-89ab-cdef-0123-456789abcdef}\\b.txt";
    EXPECT_EQ(strip_os_prefix(volume).native(), volume);
    EXPECT_EQ(strip_os_prefix(L"C:\\a").native(), L"C:\\a");

    std::filesystem::path long_path = L"C:\\" + std::wstring(300, L'x');
    EXPECT_EQ(strip_os_prefix(os_path(long_path)), long_path);
}

TEST(LongPath, PlainLongPathsWorkOnlyIfEnabled)
{
    auto dir = LongDir();
    std::ofstream plain(dir / "plain.txt");
    EXPECT_EQ(plain.is_open(), long_paths_enabled());

    std::error_code ec;
    std::filesystem::create_directories(dir / "plain_dir", ec);
    if (!long_paths_enabled())
    {
        EXPECT_TRUE(is_path_too_long_error(ec, dir / "plain_dir")) << ec.value() << " " << ec.message();
    }
    else
    {
        EXPECT_FALSE(ec) << ec.message();
    }

    // A missing file at a long path is not a path that is too long once it gets the prefix.
    std::ifstream missing(os_path(dir / "missing.txt"));
    EXPECT_FALSE(missing.is_open());
    std::filesystem::file_size(os_path(dir / "missing.txt"), ec);
    EXPECT_FALSE(is_path_too_long_error(ec, os_path(dir / "missing.txt")));
}
#endif

TEST(LongPath, IsPathTooLongError)
{
    auto short_path = std::filesystem::temp_directory_path() / "file.txt";
    EXPECT_TRUE(is_path_too_long_error(std::make_error_code(std::errc::filename_too_long), short_path));
    EXPECT_FALSE(is_path_too_long_error(std::make_error_code(std::errc::no_such_file_or_directory), short_path));
    EXPECT_FALSE(is_path_too_long_error(std::error_code(), short_path));

    std::error_code ec;
    std::filesystem::file_size(TestDir() / "missing.txt", ec);
    EXPECT_TRUE(ec);
    EXPECT_FALSE(is_path_too_long_error(ec, TestDir() / "missing.txt"));
}

TEST(LongPath, Files)
{
    auto dir            = LongDir();
    std::string content = "first line\nsecond line\n";

    File::saveFileBinary(dir / "file.txt", content.data(), content.size());
    auto binary = File::loadFileBinary(dir / "file.txt");
    EXPECT_EQ(std::string(binary.begin(), binary.end()), content);
    EXPECT_EQ(File::loadFileString(dir / "file.txt"), content);
    EXPECT_EQ(File::loadFileStringArray(dir / "file.txt"), (std::vector<std::string>{"first line", "second line"}));

    Directory directory(dir);
    EXPECT_TRUE(directory.existsFile("file.txt"));
}

TEST(LongPath, ReplaceFile)
{
    auto dir = LongDir();
    WriteText(dir / "from.txt", "new");
    WriteText(dir / "to.txt", "old");

    std::error_code ec;
    EXPECT_TRUE(replace_file(dir / "from.txt", dir / "to.txt", ec)) << ec.message();
    EXPECT_EQ(File::loadFileString(dir / "to.txt"), "new\n");
    EXPECT_FALSE(std::filesystem::exists(os_path(dir / "from.txt")));

    // From the temp folder, which can be on another drive than the target.
    auto temp = TestDir() / "from.txt";
    WriteText(temp, "moved");
    EXPECT_TRUE(replace_file(temp, dir / "to.txt", ec)) << ec.message();
    EXPECT_EQ(File::loadFileString(dir / "to.txt"), "moved\n");
}

TEST(LongPath, WriteViaShortPath)
{
    auto dir = LongDir();
    // A writer that can only take short paths, as Windows' shell and media functions.
    auto short_only_writer = [](const std::filesystem::path& path)
    {
#if defined(_WIN32)
        EXPECT_LT(path.native().size(), size_t(260));  // MAX_PATH
#endif
        std::ofstream file(path, std::ios::binary);
        file << "written";
        return file.good();
    };

    EXPECT_TRUE(write_via_short_path(dir / "file.mp4", short_only_writer));
    EXPECT_EQ(File::loadFileString(dir / "file.mp4"), "written\n");
    EXPECT_TRUE(write_via_short_path(os_path(dir / "prefixed.mp4"), short_only_writer));
    EXPECT_EQ(File::loadFileString(dir / "prefixed.mp4"), "written\n");

    auto short_target = TestDir() / "short.txt";
    EXPECT_TRUE(write_via_short_path(short_target, short_only_writer));
    EXPECT_EQ(File::loadFileString(short_target), "written\n");
    std::filesystem::remove(short_target);

    // A failed writer leaves nothing behind.
    EXPECT_FALSE(write_via_short_path(dir / "failed.mp4", [](const std::filesystem::path&) { return false; }));
    EXPECT_FALSE(std::filesystem::exists(os_path(dir / "failed.mp4")));
}

TEST(LongPath, Images)
{
    auto dir = LongDir();
    TemplatedImage<ucvec3> image(16, 24);
    for (int y = 0; y < image.h; ++y)
    {
        for (int x = 0; x < image.w; ++x)
        {
            image(y, x) = ucvec3(uint8_t(x * 10), uint8_t(y * 10), 7);
        }
    }

    std::vector<std::string> extensions = {".saigai", ".jpg"};
#ifdef SAIGA_USE_PNG
    extensions.push_back(".png");
#endif
#ifdef SAIGA_USE_LIBTIFF
    extensions.push_back(".tif");
#endif
    for (const auto& extension : extensions)
    {
        auto path = dir / ("image" + extension);
        ASSERT_TRUE(image.save(path)) << extension;
        Image loaded;
        ASSERT_TRUE(loaded.load(path)) << extension;
        EXPECT_EQ(loaded.h, image.h) << extension;
        EXPECT_EQ(loaded.w, image.w) << extension;
    }
}

#ifdef SAIGA_USE_LIBZIP
TEST(LongPath, Zip)
{
    auto dir         = LongDir();
    auto path        = dir / "archive.zip";
    std::string data = "entry data";

    WriteText(dir / "file.txt", "file");
    std::filesystem::create_directories(os_path(dir / "folder" / "nested"));
    WriteText(dir / "folder" / "nested" / "nested.txt", "nested");
    {
        ZipArchive archive(path, ZipMode::Write);
        ASSERT_TRUE(archive.is_open()) << archive.get_last_error();
        EXPECT_TRUE(archive.add_file("data.bin", data.data(), data.size(), ZipCompressionMethod::ZLIB));
        EXPECT_TRUE(archive.add_file("file.txt", dir / "file.txt", ZipCompressionMethod::ZLIB));
        EXPECT_TRUE(archive.add_folder("folder", dir / "folder", ZipCompressionMethod::ZLIB));
        EXPECT_TRUE(archive.close());
    }
    {
        // Updating writes to a temporary file next to the archive.
        ZipArchive archive(path, ZipMode::Update);
        ASSERT_TRUE(archive.is_open()) << archive.get_last_error();
        EXPECT_TRUE(archive.add_file("update.bin", data.data(), data.size(), ZipCompressionMethod::ZSTD));
        EXPECT_TRUE(archive.close());
    }

    ZipArchive archive(path, ZipMode::Read);
    ASSERT_TRUE(archive.is_open()) << archive.get_last_error();
    EXPECT_EQ(archive.file_count(), 4);
    EXPECT_TRUE(archive.find_file("file.txt").first);
    EXPECT_TRUE(archive.find_file("folder/nested/nested.txt").first);
    auto [found, file] = archive.find_file("update.bin");
    ASSERT_TRUE(found);
    std::string read(file.uncompressed_size, '\0');
    EXPECT_TRUE(file.read_all(read.data()));
    EXPECT_EQ(read, data);

    // Only the archive and the added files, no temporary file left over.
    int files = 0;
    for (auto& entry : std::filesystem::directory_iterator(os_path(dir)))
    {
        files += entry.is_regular_file();
    }
    EXPECT_EQ(files, 2);
}

TEST(LongPath, ZipFolderBelowShortFolder)
{
    // The folder itself is short, the files in it are not.
    auto dir = TestDir() / "short";
    std::error_code ec;
    std::filesystem::remove_all(os_path(dir, LongPathPrefix::Always), ec);
    auto nested = dir / std::string(150, 'n') / std::string(150, 'm');
    ASSERT_TRUE(std::filesystem::create_directories(os_path(nested), ec)) << ec.message();
    WriteText(nested / "file.txt", "file");

    {
        ZipArchive archive(TestDir() / "short.zip", ZipMode::Write);
        EXPECT_TRUE(archive.add_folder("root", dir, ZipCompressionMethod::ZLIB)) << archive.get_last_error();
        EXPECT_TRUE(archive.close());
    }
    ZipArchive archive(TestDir() / "short.zip", ZipMode::Read);
    auto name = std::filesystem::path("root") / std::string(150, 'n') / std::string(150, 'm') / "file.txt";
    EXPECT_TRUE(archive.find_file(name).first);
    archive.close();
    std::filesystem::remove(TestDir() / "short.zip");
}
#endif
