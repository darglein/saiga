/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// DiskSaveFileSystem (S25): files are opened per call (no handle cache); paths through string_to_path (UTF-8).

#include "saiga/engine/save/SaveFileSystem.h"

#include "saiga/core/util/string_to_path.h"
#include "saiga/engine/save/SaveDirectoryFormat.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <vector>

#include <system_error>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/file.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace Saiga
{
namespace Engine
{
bool SaveFileSystem::readAll(const std::string& path, std::vector<uint8_t>& out, std::string* error)
{
    uint64_t size = 0;
    if (!fileSize(path, size, error)) return false;
    // Untrusted: no file the save directory reads whole (header.sgh, the world files) is larger
    if (size > kMaxWorldFileBytes)
    {
        if (error) *error = path + ": " + std::to_string(size) + " bytes, more than the largest save file";
        return false;
    }
    out.resize(size_t(size));
    return size == 0 || readAt(path, 0, size, out.data(), error);
}

namespace
{
bool fail(std::string* error, const std::string& message)
{
    if (error) *error = message;
    return false;
}

#ifdef _WIN32
std::string lastError()
{
    return "error " + std::to_string(GetLastError());
}
#else
std::string lastError()
{
    return std::strerror(errno);
}
#endif

class DiskSaveFileSystem final : public SaveFileSystem
{
   public:
    DiskSaveFileSystem(const std::string& root, bool sync) : root(root), doSync(sync) {}
    ~DiskSaveFileSystem() override { unlock(); }

    std::filesystem::path full(const std::string& path) const
    {
        return path.empty() ? string_to_path(root) : string_to_path(root + "/" + path);
    }

    bool exists(const std::string& path) override
    {
        // A failed lookup (a transient metadata error) answers "exists": callers treat a missing file as corruption,
        // so only a confirmed absence may say false (the following read then reports the IO error)
        std::error_code ec;
        const bool e = std::filesystem::exists(full(path), ec);
        return ec ? true : e;
    }
    bool isDirectory(const std::string& path) override
    {
        std::error_code ec;
        return std::filesystem::is_directory(full(path), ec);
    }
    bool list(const std::string& dir, std::vector<std::string>& names, std::string* error) override
    {
        names.clear();
        std::error_code ec;
        for (std::filesystem::directory_iterator it(full(dir), ec), end; !ec && it != end; it.increment(ec))
            // UTF-8 (C++17: u8string is a std::string); path_to_string gives the ANSI code page on Windows
            names.push_back(it->path().filename().u8string());
        if (ec) return fail(error, "list " + dir + ": " + ec.message());
        std::sort(names.begin(), names.end());
        return true;
    }
    bool fileSize(const std::string& path, uint64_t& size, std::string* error) override
    {
        std::error_code ec;
        size = uint64_t(std::filesystem::file_size(full(path), ec));
        if (ec) return fail(error, "size of " + path + ": " + ec.message());
        return true;
    }

#ifdef _WIN32
    HANDLE openFile(const std::string& path, DWORD access, DWORD disposition)
    {
        return CreateFileW(full(path).wstring().c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    }
    bool readAt(const std::string& path, uint64_t offset, uint64_t size, uint8_t* out, std::string* error) override
    {
        HANDLE h = openFile(path, GENERIC_READ, OPEN_EXISTING);
        if (h == INVALID_HANDLE_VALUE) return fail(error, "open " + path + ": " + lastError());
        uint64_t done = 0;
        while (done < size)
        {
            OVERLAPPED o{};
            const uint64_t at = offset + done;
            o.Offset          = DWORD(at & 0xFFFFFFFFu);
            o.OffsetHigh      = DWORD(at >> 32);
            const DWORD chunk = DWORD(std::min<uint64_t>(size - done, 1u << 30));
            DWORD got         = 0;
            const BOOL ok     = ReadFile(h, out + done, chunk, &got, &o);
            if (!ok || got == 0)
            {
                // GetLastError before CloseHandle (which may change it); a successful 0-byte read is the end of the
                // file
                const std::string why = ok ? std::string("unexpected end of file") : lastError();
                CloseHandle(h);
                return fail(error, "read " + path + ": " + why);
            }
            done += got;
        }
        CloseHandle(h);
        return true;
    }
    bool writeAll(HANDLE h, const uint8_t* data, size_t size)
    {
        size_t done = 0;
        while (done < size)
        {
            DWORD n           = 0;
            const DWORD chunk = DWORD(std::min<size_t>(size - done, size_t(1) << 30));
            if (!WriteFile(h, data + done, chunk, &n, nullptr)) return false;
            done += n;
        }
        return true;
    }
    bool writeNew(const std::string& path, const uint8_t* data, size_t size, std::string* error) override
    {
        HANDLE h = openFile(path, GENERIC_WRITE, CREATE_ALWAYS);
        if (h == INVALID_HANDLE_VALUE) return fail(error, "create " + path + ": " + lastError());
        const bool ok         = writeAll(h, data, size);
        const std::string why = ok ? std::string() : lastError();
        CloseHandle(h);
        return ok || fail(error, "write " + path + ": " + why);
    }
    bool append(const std::string& path, const uint8_t* data, size_t size, std::string* error) override
    {
        HANDLE h = openFile(path, FILE_APPEND_DATA, OPEN_ALWAYS);
        if (h == INVALID_HANDLE_VALUE) return fail(error, "open " + path + ": " + lastError());
        const bool ok         = writeAll(h, data, size);
        const std::string why = ok ? std::string() : lastError();
        CloseHandle(h);
        return ok || fail(error, "append " + path + ": " + why);
    }
    bool truncate(const std::string& path, uint64_t size, std::string* error) override
    {
        HANDLE h = openFile(path, GENERIC_WRITE, OPEN_EXISTING);
        if (h == INVALID_HANDLE_VALUE) return fail(error, "open " + path + ": " + lastError());
        LARGE_INTEGER li;
        li.QuadPart           = LONGLONG(size);
        const bool ok         = SetFilePointerEx(h, li, nullptr, FILE_BEGIN) && SetEndOfFile(h);
        const std::string why = ok ? std::string() : lastError();
        CloseHandle(h);
        return ok || fail(error, "truncate " + path + ": " + why);
    }
    bool syncFile(const std::string& path, std::string* error) override
    {
        if (!doSync) return true;
        HANDLE h = openFile(path, GENERIC_WRITE, OPEN_EXISTING);
        if (h == INVALID_HANDLE_VALUE) return fail(error, "open " + path + ": " + lastError());
        const bool ok         = FlushFileBuffers(h);
        const std::string why = ok ? std::string() : lastError();
        CloseHandle(h);
        return ok || fail(error, "sync " + path + ": " + why);
    }
    // NTFS orders the metadata of a rename (MOVEFILE_WRITE_THROUGH): no directory sync
    bool syncDirectory(const std::string&, std::string*) override { return true; }
    bool rename(const std::string& from, const std::string& to, std::string* error) override
    {
        if (!MoveFileExW(full(from).wstring().c_str(), full(to).wstring().c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            return fail(error, "rename " + from + " to " + to + ": " + lastError());
        return true;
    }
    bool lock(std::string* error) override
    {
        if (lockHandle != INVALID_HANDLE_VALUE) return true;
        std::error_code ec;
        std::filesystem::create_directories(full(""), ec);
        HANDLE h =
            CreateFileW(full("lock").wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return fail(error, "the lock file of " + root + ": " + lastError());
        OVERLAPPED o{};
        if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &o))
        {
            CloseHandle(h);
            return fail(error, "the save directory " + root + " is in use (locked by another world or process)");
        }
        lockHandle = h;
        return true;
    }
    void unlock() override
    {
        if (lockHandle == INVALID_HANDLE_VALUE) return;
        OVERLAPPED o{};
        UnlockFileEx(lockHandle, 0, 1, 0, &o);
        CloseHandle(lockHandle);
        lockHandle = INVALID_HANDLE_VALUE;
    }
#else
    bool readAt(const std::string& path, uint64_t offset, uint64_t size, uint8_t* out, std::string* error) override
    {
        const int fd = ::open(full(path).c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return fail(error, "open " + path + ": " + lastError());
        uint64_t done = 0;
        while (done < size)
        {
            const ssize_t n = ::pread(fd, out + done, size_t(size - done), off_t(offset + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0)
            {
                const std::string why = n == 0 ? std::string("unexpected end of file") : lastError();
                ::close(fd);
                return fail(error, "read " + path + ": " + why);
            }
            done += uint64_t(n);
        }
        ::close(fd);
        return true;
    }
    bool writeFd(int fd, const uint8_t* data, size_t size)
    {
        size_t done = 0;
        while (done < size)
        {
            const ssize_t n = ::write(fd, data + done, size - done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            done += size_t(n);
        }
        return true;
    }
    bool writeWith(const std::string& path, int flags, const uint8_t* data, size_t size, std::string* error)
    {
        const int fd = ::open(full(path).c_str(), flags | O_CLOEXEC, 0644);
        if (fd < 0) return fail(error, "open " + path + ": " + lastError());
        const bool ok         = writeFd(fd, data, size);
        const std::string why = ok ? std::string() : lastError();
        ::close(fd);
        return ok || fail(error, "write " + path + ": " + why);
    }
    bool writeNew(const std::string& path, const uint8_t* data, size_t size, std::string* error) override
    {
        return writeWith(path, O_WRONLY | O_CREAT | O_TRUNC, data, size, error);
    }
    bool append(const std::string& path, const uint8_t* data, size_t size, std::string* error) override
    {
        return writeWith(path, O_WRONLY | O_CREAT | O_APPEND, data, size, error);
    }
    bool truncate(const std::string& path, uint64_t size, std::string* error) override
    {
        if (::truncate(full(path).c_str(), off_t(size)) != 0)
            return fail(error, "truncate " + path + ": " + lastError());
        return true;
    }
    bool syncPath(const std::string& path, int flags, std::string* error)
    {
        if (!doSync) return true;
        const int fd = ::open(full(path).c_str(), flags | O_CLOEXEC);
        if (fd < 0) return fail(error, "open " + path + ": " + lastError());
        const bool ok         = ::fsync(fd) == 0;
        const std::string why = ok ? std::string() : lastError();
        ::close(fd);
        return ok || fail(error, "sync " + path + ": " + why);
    }
    bool syncFile(const std::string& path, std::string* error) override { return syncPath(path, O_RDONLY, error); }
    bool syncDirectory(const std::string& dir, std::string* error) override
    {
        return syncPath(dir, O_RDONLY | O_DIRECTORY, error);
    }
    bool rename(const std::string& from, const std::string& to, std::string* error) override
    {
        if (::rename(full(from).c_str(), full(to).c_str()) != 0)
            return fail(error, "rename " + from + " to " + to + ": " + lastError());
        return true;
    }
    bool lock(std::string* error) override
    {
        if (lockFd >= 0) return true;
        std::error_code ec;
        std::filesystem::path dir = std::filesystem::absolute(std::filesystem::path(full("")), ec).lexically_normal();
        if (ec) return fail(error, "the absolute path of " + root + ": " + ec.message());
        if (!dir.has_filename()) dir = dir.parent_path();
        std::filesystem::create_directories(dir, ec);
        if (doSync)
        {
            // Every entry on the path to the save directory is made durable at each open: an ancestor that exists
            // may have been created by an earlier attempt (or another process) whose sync failed or has not run yet,
            // so existence is no proof of durability (one fsync per ancestor directory)
            for (std::filesystem::path p = dir; p.has_relative_path(); p = p.parent_path())
            {
                const std::filesystem::path parent = p.parent_path();
                const int d                        = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
                if (d < 0) return fail(error, "open the directory " + parent.string() + ": " + lastError());
                const bool synced     = ::fsync(d) == 0;
                const std::string why = synced ? std::string() : lastError();
                ::close(d);
                if (!synced) return fail(error, "sync the directory " + parent.string() + ": " + why);
            }
        }
        const int fd = ::open(full("lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) return fail(error, "the lock file of " + root + ": " + lastError());
        // flock locks the open file description: a second open in this process conflicts as well
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0)
        {
            ::close(fd);
            return fail(error, "the save directory " + root + " is in use (locked by another world or process)");
        }
        lockFd = fd;
        return true;
    }
    void unlock() override
    {
        if (lockFd < 0) return;
        ::flock(lockFd, LOCK_UN);
        ::close(lockFd);
        lockFd = -1;
    }
#endif

    bool makeDirectory(const std::string& dir, std::string* error) override
    {
        std::error_code ec;
        std::filesystem::create_directory(full(dir), ec);
        if (ec) return fail(error, "create directory " + dir + ": " + ec.message());
        return true;
    }
    bool remove(const std::string& path, std::string* error) override
    {
        std::error_code ec;
        std::filesystem::remove(full(path), ec);
        if (ec) return fail(error, "remove " + path + ": " + ec.message());
        return true;
    }
    std::string describe() const override { return root; }

   private:
    std::string root;
    bool doSync = true;
#ifdef _WIN32
    HANDLE lockHandle = INVALID_HANDLE_VALUE;
#else
    int lockFd = -1;
#endif
};
}  // namespace

std::shared_ptr<SaveFileSystem> diskFileSystem(const std::string& root, bool sync)
{
    return std::make_shared<DiskSaveFileSystem>(root, sync);
}

}  // namespace Engine
}  // namespace Saiga
