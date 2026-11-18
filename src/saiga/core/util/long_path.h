/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <filesystem>
#include <functional>

#include <system_error>

// Paths longer than MAX_PATH (260 characters) on Windows.
//
// Windows' file functions take longer paths only if the process is long path aware (its manifest has
// longPathAware, see saiga_enable_long_paths in cmake/saiga_long_paths.cmake; MSVC and clang-cl builds only) AND the
// system enables them
// (LongPathsEnabled in the registry, off by default), or if the path has the \\?\ prefix. os_path() adds that prefix,
// so a path works on every system. Keep paths without the prefix everywhere else (in files, in the UI, in comparisons)
// and call os_path() only where a path is handed to the file system:
//
//     std::ofstream file(Saiga::os_path(path), std::ios::binary);
//     std::filesystem::create_directories(Saiga::os_path(dir), ec);
//
// Call it on the full path: os_path(dir) / "name" is not prefixed if dir is short but dir / "name" is not.
// std::filesystem, fstream, _wfopen and the *W functions of Win32 accept prefixed paths. Shell functions
// (ShellExecute, file dialogs, Explorer), CreateProcess's working directory, LoadLibrary and libraries that take narrow
// (char) paths don't: use write_via_short_path() to write a file through them.
//
// Everywhere else (Linux, macOS) these functions change nothing.
namespace Saiga
{
enum class LongPathPrefix
{
    // Only paths too long for MAX_PATH get the prefix. Shorter paths are returned unchanged, also relative ones.
    IfNeeded,
    // Every path gets the prefix. Use it for a folder that is iterated recursively or below which long paths are
    // created: a short folder can contain paths that are too long.
    Always,
};

// The path to hand to the file system: on Windows, absolute and normalized ('/' becomes '\', "." and ".." are
// resolved) with the \\?\ or \\?\UNC\ prefix, if it needs one. Paths that already have a prefix are returned unchanged.
SAIGA_CORE_API std::filesystem::path os_path(const std::filesystem::path& path,
                                             LongPathPrefix prefix = LongPathPrefix::IfNeeded);

// Removes the \\?\ or \\?\UNC\ prefix from a path of a drive or a network share, for example one returned by a
// directory iterator over os_path(dir, LongPathPrefix::Always). Other paths are returned unchanged.
SAIGA_CORE_API std::filesystem::path strip_os_prefix(const std::filesystem::path& path);

// True if this process can use paths longer than MAX_PATH without the \\?\ prefix: its manifest is long path aware and
// the system enables long paths. Always true outside of Windows.
SAIGA_CORE_API bool long_paths_enabled();

// True if accessing path failed with ec because the path or a file name in it is too long. On Windows this also
// includes "not found" errors for a path without the prefix that is too long for MAX_PATH, as long paths are not
// enabled: Windows reports those as not found once part of the path exists.
SAIGA_CORE_API bool is_path_too_long_error(const std::error_code& ec, const std::filesystem::path& path);

// Moves a file, replacing the target if it exists, also to another drive. On failure the target is unchanged. True
// means the target holds the file; across drives the file is copied, and a source that can't be removed afterwards
// stays.
SAIGA_CORE_API bool replace_file(const std::filesystem::path& from, const std::filesystem::path& to,
                                 std::error_code& ec);

// For writers that can't take a long path (shell and media functions, libraries with narrow paths): if the target needs
// the \\?\ prefix or already has it, write() writes to a file of the same extension in the temp folder, which is then
// moved to the target and gets the permissions of the target's folder. Otherwise write() writes to the target
// directly. Returns false if write() returns false, the temp folder is too long itself or the file can't be moved; the
// temporary file is removed in all cases.
SAIGA_CORE_API bool write_via_short_path(const std::filesystem::path& target,
                                         const std::function<bool(const std::filesystem::path&)>& write);
}  // namespace Saiga
