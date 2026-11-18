/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/long_path.h"

#include <atomic>
#include <cwctype>
#include <random>
#include <string>

#if defined(_WIN32)
#    include <windows.h>
#endif

namespace Saiga
{
#if defined(_WIN32)
static const std::wstring verbatim_prefix     = LR"(\\?\)";
static const std::wstring verbatim_unc_prefix = LR"(\\?\UNC\)";

// CreateDirectoryW takes at most MAX_PATH - 12 characters (room for an 8.3 file name), other functions MAX_PATH - 1.
static constexpr size_t max_unprefixed_length = MAX_PATH - 12 - 1;

static bool is_separator(wchar_t c)
{
    return c == L'\\' || c == L'/';
}

// \\?\ (verbatim), \\.\ (device) and \??\ (NT object namespace) paths are passed to the OS as they are.
static bool has_os_prefix(const std::wstring& s)
{
    if (s.size() < 4 || !is_separator(s[0]) || !is_separator(s[3])) return false;
    return (is_separator(s[1]) && (s[2] == L'?' || s[2] == L'.')) || (s[1] == L'?' && s[2] == L'?');
}
#endif

std::filesystem::path os_path(const std::filesystem::path& path, LongPathPrefix prefix)
{
#if defined(_WIN32)
    if (path.empty() || has_os_prefix(path.native())) return path;

    std::error_code ec;
    std::filesystem::path absolute = std::filesystem::absolute(path, ec);
    if (ec) return path;

    const std::wstring& s = absolute.native();
    if (prefix == LongPathPrefix::IfNeeded && s.size() <= max_unprefixed_length) return path;

    if (s.size() >= 2 && is_separator(s[0]) && is_separator(s[1]))
    {
        // \\server\share\... becomes \\?\UNC\server\share\...
        return verbatim_unc_prefix + s.substr(2);
    }
    if (s.size() >= 3 && s[1] == L':' && is_separator(s[2]))
    {
        return verbatim_prefix + s;
    }
    return path;
#else
    (void)prefix;
    return path;
#endif
}

std::filesystem::path strip_os_prefix(const std::filesystem::path& path)
{
#if defined(_WIN32)
    const std::wstring& s = path.native();
    if (s.compare(0, verbatim_unc_prefix.size(), verbatim_unc_prefix) == 0)
    {
        return LR"(\\)" + s.substr(verbatim_unc_prefix.size());
    }
    if (s.compare(0, verbatim_prefix.size(), verbatim_prefix) == 0)
    {
        std::wstring rest = s.substr(verbatim_prefix.size());
        // Only drive paths: \\?\Volume{...}\ and the like have no form without the prefix.
        if (rest.size() >= 2 && iswalpha(rest[0]) && rest[1] == L':') return rest;
    }
#endif
    return path;
}

bool long_paths_enabled()
{
#if defined(_WIN32)
    static const bool enabled = []
    {
        // Windows 10 1607 and newer: true if the manifest is long path aware and LongPathsEnabled is set.
        using RtlAreLongPathsEnabledFn = BOOLEAN(WINAPI*)();
        HMODULE ntdll                  = GetModuleHandleW(L"ntdll.dll");
        if (!ntdll) return false;
        auto fn = reinterpret_cast<RtlAreLongPathsEnabledFn>(
            reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlAreLongPathsEnabled")));
        return fn && fn() != 0;
    }();
    return enabled;
#else
    return true;
#endif
}

bool is_path_too_long_error(const std::error_code& ec, const std::filesystem::path& path)
{
    if (!ec) return false;
    if (ec == std::errc::filename_too_long) return true;
#if defined(_WIN32)
    if (ec.category() != std::system_category()) return false;
    if (ec.value() == ERROR_FILENAME_EXCED_RANGE) return true;
    // Below an existing folder, Windows reports a path that is too long as not found.
    bool not_found = ec.value() == ERROR_PATH_NOT_FOUND || ec.value() == ERROR_FILE_NOT_FOUND;
    return not_found && !long_paths_enabled() && !path.empty() && !has_os_prefix(path.native()) &&
           os_path(path).native() != path.native();
#else
    (void)path;
    return false;
#endif
}

bool replace_file(const std::filesystem::path& from, const std::filesystem::path& to, std::error_code& ec)
{
    ec.clear();
#if defined(_WIN32)
    if (!MoveFileExW(os_path(from).c_str(), os_path(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED))
    {
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category());
        return false;
    }
    return true;
#else
    std::filesystem::rename(from, to, ec);
    if (ec == std::errc::cross_device_link)
    {
        ec.clear();
        if (!std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec)) return false;
        std::filesystem::remove(from, ec);
        // The target is complete; a source that can't be removed is not an error of the move.
        ec.clear();
    }
    return !ec;
#endif
}

bool write_via_short_path(const std::filesystem::path& target,
                          const std::function<bool(const std::filesystem::path&)>& write)
{
    const std::filesystem::path plain_target = strip_os_prefix(target);
    if (os_path(plain_target).native() == plain_target.native())
    {
        return write(plain_target);
    }

    std::error_code ec;
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path(ec);
    if (ec) return false;

    static std::atomic<uint64_t> counter{0};
    std::random_device random;
    std::filesystem::path temp;
    for (int attempt = 0; attempt < 16; ++attempt)
    {
        std::string name = "saiga_" + std::to_string(random()) + "_" + std::to_string(counter++);
        temp             = temp_dir / std::filesystem::path(name).concat(target.extension().native());
        if (!std::filesystem::exists(temp, ec)) break;
        temp.clear();
    }
    if (temp.empty()) return false;

    bool written = false;
    try
    {
        written = write(temp);
    }
    catch (...)
    {
        std::filesystem::remove(temp, ec);
        throw;
    }

    if (!written || !replace_file(temp, target, ec))
    {
        std::filesystem::remove(temp, ec);
        return false;
    }
    return true;
}
}  // namespace Saiga
