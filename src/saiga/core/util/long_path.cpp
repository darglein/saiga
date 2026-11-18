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
// SetNamedSecurityInfoW; after windows.h
#    include <aclapi.h>
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
    // Normalizing only shortens a path, so a short absolute path needs neither the prefix nor GetFullPathNameW.
    if (prefix == LongPathPrefix::IfNeeded && path.is_absolute() && path.native().size() <= max_unprefixed_length)
    {
        return path;
    }

    std::error_code ec;
    std::filesystem::path absolute = std::filesystem::absolute(path, ec);
    if (ec) return path;
    // Windows doesn't normalize a path after the prefix, and libstdc++'s absolute() returns absolute paths as they are.
    absolute = absolute.lexically_normal().make_preferred();

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
    const bool system = ec.category() == std::system_category();
    if (system && ec.value() == ERROR_FILENAME_EXCED_RANGE) return true;

    // Below an existing folder, Windows reports a path that is too long as not found (libstdc++ as ENOENT).
    const bool not_found = ec == std::errc::no_such_file_or_directory || (system && ec.value() == ERROR_PATH_NOT_FOUND);
    if (!not_found || long_paths_enabled() || path.empty() || has_os_prefix(path.native())) return false;
    // Only paths too long for the file functions: a shorter path that isn't found is missing.
    std::error_code absolute_ec;
    const std::filesystem::path absolute = std::filesystem::absolute(path, absolute_ec);
    return !absolute_ec && absolute.native().size() >= MAX_PATH;
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
    if (ec != std::errc::cross_device_link) return !ec;

    // Copy next to the target and rename it over the target: copying over it directly would destroy it if the copy
    // fails, for example when the disk is full.
    ec.clear();
    std::error_code ignored;
    std::filesystem::path staging = to;
    staging += ".saiga_" + std::to_string(std::random_device()());
    if (!std::filesystem::copy_file(from, staging, std::filesystem::copy_options::overwrite_existing, ec))
    {
        std::filesystem::remove(staging, ignored);
        return false;
    }
    std::filesystem::rename(staging, to, ec);
    if (ec)
    {
        std::filesystem::remove(staging, ignored);
        return false;
    }
    std::filesystem::remove(from, ignored);
    return true;
#endif
}

bool write_via_short_path(const std::filesystem::path& target,
                          const std::function<bool(const std::filesystem::path&)>& write)
{
#if defined(_WIN32)
    // A prefixed target goes through a temporary file too: without the prefix, Windows may resolve it to another
    // file (it drops trailing dots and spaces) or not at all (volume and device paths).
    const bool prefixed = has_os_prefix(target.native());
#else
    const bool prefixed = false;
#endif
    if (!prefixed && os_path(target).native() == target.native())
    {
        return write(target);
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
    // In a temp folder that is too long itself, the writer can't take the temporary file either.
    if (temp.empty() || os_path(temp).native() != temp.native()) return false;

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
#if defined(_WIN32)
    // A file moved within a drive keeps the permissions it got in the temp folder, which are often private to the
    // user. An empty, unprotected DACL makes it inherit those of its new folder, as a file written there directly.
    ACL empty_acl;
    std::wstring target_path = os_path(target).native();
    if (InitializeAcl(&empty_acl, sizeof(empty_acl), ACL_REVISION))
    {
        SetNamedSecurityInfoW(target_path.data(), SE_FILE_OBJECT,
                              DACL_SECURITY_INFORMATION | UNPROTECTED_DACL_SECURITY_INFORMATION, nullptr, nullptr,
                              &empty_acl, nullptr);
    }
#endif
    return true;
}
}  // namespace Saiga
