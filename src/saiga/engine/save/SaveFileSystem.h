/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SaveFileSystem (docs/engine_design.md §15.1; S25): every IO of a save directory goes through this interface, so a
// test can stop it after any mutating operation (MemorySaveFileSystem's crash model, G9). Paths are relative to the
// save directory, '/' separated, without "." or ".." components ("" is the directory itself). Every call returns false
// with *error on failure (error may be null).

#pragma once

#include "saiga/export.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API SaveFileSystem
{
   public:
    virtual ~SaveFileSystem()                    = default;
    virtual bool exists(const std::string& path) = 0;
    // The names in a directory (files and directories, sorted)
    virtual bool list(const std::string& dir, std::vector<std::string>& names, std::string* error)                 = 0;
    virtual bool isDirectory(const std::string& path)                                                              = 0;
    virtual bool fileSize(const std::string& path, uint64_t& size, std::string* error)                             = 0;
    virtual bool readAt(const std::string& path, uint64_t offset, uint64_t size, uint8_t* out, std::string* error) = 0;
    // ---- mutating operations (counted by the memory file system)
    // One level: the parent must exist; an existing directory is not an error
    virtual bool makeDirectory(const std::string& dir, std::string* error) = 0;
    // writeNew creates or truncates; append creates a missing file
    virtual bool writeNew(const std::string& path, const uint8_t* data, size_t size, std::string* error) = 0;
    virtual bool append(const std::string& path, const uint8_t* data, size_t size, std::string* error)   = 0;
    virtual bool truncate(const std::string& path, uint64_t size, std::string* error)                    = 0;
    virtual bool syncFile(const std::string& path, std::string* error)                                   = 0;
    virtual bool syncDirectory(const std::string& dir, std::string* error)                               = 0;
    // An atomic replace of `to`
    virtual bool rename(const std::string& from, const std::string& to, std::string* error) = 0;
    virtual bool remove(const std::string& path, std::string* error)                        = 0;
    // An exclusive lock of the directory (the file <dir>/lock, created if missing), released by unlock or when the
    // process ends; the directory is created if missing (its parent must exist)
    virtual bool lock(std::string* error) = 0;
    virtual void unlock()                 = 0;
    // A description for messages (the directory path)
    virtual std::string describe() const = 0;

    // Reads a whole file
    bool readAll(const std::string& path, std::vector<uint8_t>& out, std::string* error);
};

// The directory root on disk (POSIX: open/pwrite/pread/fsync/rename/flock; Windows: CreateFileW, FlushFileBuffers,
// MoveFileExW with MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH, LockFileEx; a directory sync is a no-op there).
// sync = false makes syncFile and syncDirectory no-ops (tests on a real disk)
SAIGA_ENGINE_API std::shared_ptr<SaveFileSystem> diskFileSystem(const std::string& root, bool sync = true);

}  // namespace Engine
}  // namespace Saiga
