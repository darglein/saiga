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

#if defined(_WIN32)
#    include <windows.h>
// After windows.h
#    include <aclapi.h>
#    include <sddl.h>
#endif

// Built twice on Windows: test_core_long_path is not long path aware, so it behaves like a system without
// LongPathsEnabled, and test_core_long_path_aware has the manifest of saiga_enable_long_paths. Both must pass.

using namespace Saiga;

#if defined(_WIN32)
static const std::wstring verbatim = LR"(\\?\)";
// A short absolute path, which needs no prefix however long the temp folder is.
static const std::filesystem::path short_absolute = LR"(C:\folder\file.txt)";
#else
static const std::filesystem::path short_absolute = "/folder/file.txt";
#endif

// A short folder of this process in the temp folder, removed at exit: ctest runs both executables at the same time.
struct ProcessDir
{
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                ("saiga_test_core_long_path_" + std::to_string(std::random_device()()));
    ProcessDir() { std::filesystem::create_directories(os_path(dir)); }
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
    // Unless the working directory is so long that the relative path resolves to a long one.
    auto relative = std::filesystem::path("folder") / "file.txt";
    if (std::filesystem::absolute(relative).native().size() < 240)
    {
        EXPECT_EQ(os_path(relative), relative);
    }

    EXPECT_EQ(os_path(short_absolute), short_absolute);
    EXPECT_EQ(os_path(std::filesystem::path()), std::filesystem::path());
}

#if defined(_WIN32)
TEST(LongPath, LongPathsGetThePrefix)
{
    std::wstring long_name(300, L'x');
    auto path = os_path(std::filesystem::path(L"C:/folder/../other/" + long_name));
    EXPECT_EQ(path.native(), verbatim + L"C:\\other\\" + long_name);

    // The server and share stay; only what follows them is normalized.
    EXPECT_EQ(os_path(std::filesystem::path(L"//server/share/a/../" + long_name)).native(),
              verbatim + L"UNC\\server\\share\\" + long_name);
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
    const auto& short_path = short_absolute;
    EXPECT_TRUE(is_path_too_long_error(std::make_error_code(std::errc::filename_too_long), short_path));
    EXPECT_FALSE(is_path_too_long_error(std::make_error_code(std::errc::no_such_file_or_directory), short_path));
    EXPECT_FALSE(is_path_too_long_error(std::error_code(), short_path));

    // A missing file that the OS gets in a form it can open (prefixed if needed, as the temp folder can be long).
    std::error_code ec;
    const auto missing_file = os_path(TestDir() / "missing.txt");
    (void)std::filesystem::file_size(missing_file, ec);
    EXPECT_TRUE(ec);
    EXPECT_FALSE(is_path_too_long_error(ec, missing_file));

    // Short enough for the file functions, even if os_path would prefix it: a missing file, not a path too long.
    auto dir = TestDir() / "folder";
    if (dir.native().size() + 2 > 250) GTEST_SKIP() << "The temp folder is too long for this case: " << dir;
    std::filesystem::create_directories(dir);
    auto missing = dir / std::string(250 - (dir.native().size() + 1), 'm');
    ASSERT_EQ(missing.native().size(), 250u);
    (void)std::filesystem::file_size(missing, ec);
    EXPECT_TRUE(ec);
    EXPECT_FALSE(is_path_too_long_error(ec, missing));
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

    // On Windows, a short target with the prefix is written through the temp folder too: without the prefix it could
    // name another file. Elsewhere os_path adds no prefix and the target is written directly.
    auto prefixed_short = os_path(TestDir() / "prefixed_short.txt", LongPathPrefix::Always);
    std::filesystem::path written_to;
    EXPECT_TRUE(write_via_short_path(prefixed_short,
                                     [&](const std::filesystem::path& path)
                                     {
                                         written_to = path;
                                         return short_only_writer(path);
                                     }));
#if defined(_WIN32)
    EXPECT_NE(written_to.parent_path(), TestDir());
#else
    EXPECT_EQ(written_to, prefixed_short);
#endif
    EXPECT_EQ(File::loadFileString(TestDir() / "prefixed_short.txt"), "written\n");

    // A failed writer leaves nothing behind.
    EXPECT_FALSE(write_via_short_path(dir / "failed.mp4", [](const std::filesystem::path&) { return false; }));
    EXPECT_FALSE(std::filesystem::exists(os_path(dir / "failed.mp4")));
}

#if defined(_WIN32)
TEST(LongPath, WriteViaShortPathRefusesPrefixedTempFolder)
{
    // A temp folder with the prefix can't be handed to a writer that only takes short paths.
    auto dir                 = LongDir();
    const wchar_t* tmp       = _wgetenv(L"TMP");
    const std::wstring saved = tmp ? tmp : L"";
    ASSERT_EQ(_wputenv_s(L"TMP", (verbatim + std::filesystem::temp_directory_path().native()).c_str()), 0);
    std::error_code ec;
    const auto temp_dir = std::filesystem::temp_directory_path(ec);

    bool called  = false;
    bool written = write_via_short_path(dir / "file.mp4",
                                        [&](const std::filesystem::path&)
                                        {
                                            called = true;
                                            return true;
                                        });
    _wputenv_s(L"TMP", saved.c_str());

    if (temp_dir.native().rfind(verbatim, 0) != 0) GTEST_SKIP() << "The temp folder lost its prefix: " << temp_dir;
    EXPECT_FALSE(written);
    EXPECT_FALSE(called);
}

// The access control entries of a file's DACL in SDDL, without the descriptor's control flags.
static std::wstring AcesOf(const std::filesystem::path& file)
{
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    std::wstring path               = os_path(file).native();
    if (GetNamedSecurityInfoW(path.data(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, nullptr,
                              nullptr, &descriptor) != ERROR_SUCCESS)
    {
        return L"";
    }
    LPWSTR text = nullptr;
    std::wstring result;
    if (ConvertSecurityDescriptorToStringSecurityDescriptorW(descriptor, SDDL_REVISION_1, DACL_SECURITY_INFORMATION,
                                                             &text, nullptr))
    {
        result = text;
        LocalFree(text);
    }
    LocalFree(descriptor);
    auto first_ace = result.find(L'(');
    return first_ace == std::wstring::npos ? L"" : result.substr(first_ace);
}

TEST(LongPath, WriteViaShortPathInheritsPermissions)
{
    // A file moved from the temp folder keeps the permissions it got there unless they are reset. The target folder
    // gets its own permissions (everyone, protected from those of the temp folder), so a staged file must end up with
    // those, as a file written into the folder directly.
    auto dir                        = LongDir();
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    ASSERT_TRUE(ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;FA;;;WD)", SDDL_REVISION_1,
                                                                     &descriptor, nullptr));
    BOOL present = FALSE, defaulted = FALSE;
    PACL dacl = nullptr;
    ASSERT_TRUE(GetSecurityDescriptorDacl(descriptor, &present, &dacl, &defaulted));
    std::wstring dir_path = os_path(dir).native();
    ASSERT_EQ(SetNamedSecurityInfoW(dir_path.data(), SE_FILE_OBJECT,
                                    DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, nullptr, nullptr,
                                    dacl, nullptr),
              ERROR_SUCCESS);
    LocalFree(descriptor);

    std::ofstream(os_path(dir / "direct.txt")) << "direct";
    EXPECT_TRUE(write_via_short_path(dir / "staged.txt",
                                     [](const std::filesystem::path& path)
                                     {
                                         std::ofstream file(path);
                                         file << "staged";
                                         return file.good();
                                     }));
    auto direct = AcesOf(dir / "direct.txt");
    ASSERT_FALSE(direct.empty());
    EXPECT_EQ(AcesOf(dir / "staged.txt"), direct);
}
#endif

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
