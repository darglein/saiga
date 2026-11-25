/**
 * Copyright (c) 2026 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#ifdef SAIGA_USE_ZLIB

namespace Saiga
{

// Receives the bytes of an archive in order.
struct SAIGA_CORE_API ZipStreamSink
{
    virtual ~ZipStreamSink() = default;
    // Takes all `size` bytes or throws.
    virtual void write(const void* data, size_t size) = 0;
};

// Writes to a file, which it creates or truncates. Throws std::runtime_error when it cannot.
class SAIGA_CORE_API ZipFileSink : public ZipStreamSink
{
   public:
    explicit ZipFileSink(const std::filesystem::path& path);
    void write(const void* data, size_t size) override;
    // Hands buffered bytes to the OS, so that other readers of the file see them.
    void flush();

   private:
    std::ofstream file_;
};

// Writes a zip archive front to back, with stored (uncompressed) entries only. Bytes go out in order and are never
// rewritten, so the archive can go to a pipe or into an entry of an outer archive that is being written itself.
// Entries of unknown size get a data descriptor; ZIP64 records are used where sizes, offsets or the entry count need
// them. Entry names are UTF-8 with '/' as separator; readers such as libzip reject names that are not valid UTF-8.
// Errors throw std::runtime_error (a failed sink) or std::logic_error (a call out of order); after a sink error the
// writer refuses further calls.
//
// The writer is itself a sink for the entry it has open, so an archive nests in an entry of another:
//
//    ZipStreamWriter outer("out.zip");
//    outer.begin_entry("inner.zip");
//    ZipStreamWriter inner(&outer);
//    inner.add_entry("a.txt", "hello", 5);
//    inner.finish();
//    outer.end_entry();
//    outer.finish();
//
class SAIGA_CORE_API ZipStreamWriter : public ZipStreamSink
{
   public:
    // Does not own `sink`, which must outlive the writer. A pointer, as a reference to another ZipStreamWriter
    // would pick the deleted copy constructor.
    explicit ZipStreamWriter(ZipStreamSink* sink);
    // Writes to a ZipFileSink of its own.
    explicit ZipStreamWriter(const std::filesystem::path& path);
    // Does not finish(): an archive left unfinished has no central directory.
    ~ZipStreamWriter();

    ZipStreamWriter(const ZipStreamWriter&)            = delete;
    ZipStreamWriter& operator=(const ZipStreamWriter&) = delete;

    // Adds a whole entry. Returns the offset of its first data byte.
    uint64_t add_entry(const std::string& name, const void* data, size_t size);

    // Starts an entry whose bytes follow through write(). Returns the offset of its first data byte.
    uint64_t begin_entry(const std::string& name);
    void write(const void* data, size_t size) override;
    void end_entry();

    // Writes the central directory. Nothing can be added afterwards.
    void finish();

    // Bytes written so far, which is where the next byte will lie in the archive.
    uint64_t offset() const { return offset_; }

    // Flushes the file when writing to a path, so that other readers see the bytes written so far. No-op for a sink.
    void flush();

   private:
    struct Entry
    {
        std::string name;
        uint16_t flags          = 0;
        uint32_t crc            = 0;
        uint64_t size           = 0;
        uint64_t header_offset  = 0;
        bool zip64_local_header = false;
    };

    uint64_t start_entry(const std::string& name, bool streamed, uint64_t size, uint32_t crc);
    void emit(const void* data, size_t size);
    void check_writable(const char* call) const;

    std::optional<ZipFileSink> file_;
    ZipStreamSink* sink_ = nullptr;
    uint64_t offset_     = 0;
    uint16_t dos_time_   = 0;
    uint16_t dos_date_   = 0;
    std::vector<Entry> entries_;
    bool in_entry_ = false;
    bool finished_ = false;
    bool failed_   = false;
};

}  // namespace Saiga

#endif
