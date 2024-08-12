#pragma once

#include "saiga/config.h"
#include "saiga/core/util/ProgressBar.h"
#include <string>
#include <filesystem>

#if defined(SAIGA_USE_LIBZIP)

struct zip;
struct zip_source;

namespace Saiga
{

enum class ZipMode
{
    Read,
    Write,
};

struct SAIGA_CORE_API ZipArchiveFile
{
    std::filesystem::path filename;
    size_t compressed_size;
    size_t uncompressed_size;
    zip* archive;

    bool read(void* out_data, ProgressBarManager* progress_bar = nullptr) const;
};

struct SAIGA_CORE_API ZipIncrementalWrite
{
    ZipIncrementalWrite() = default;
    ZipIncrementalWrite(zip* archive, zip_source* source) : archive(archive), source(source) {}
    ZipIncrementalWrite(ZipIncrementalWrite&& o) noexcept
        : archive(std::exchange(o.archive, nullptr)), source(std::exchange(o.source, nullptr))
    {
    }
    ~ZipIncrementalWrite();

    bool write(void* data, size_t size);

    operator bool() const { return source != nullptr; }

    zip* archive = nullptr;
    zip_source* source = nullptr;
};

struct ZipCustomSource
{
    virtual size_t total_size() const = 0;
    virtual size_t read_next(void* dest, size_t len) = 0;
};

struct SAIGA_CORE_API ZipArchive
{
    ZipArchive(const std::filesystem::path& path, ZipMode mode);
    ZipArchive(ZipArchive&& o);
    ZipArchive(const ZipArchive&) = delete;
    ~ZipArchive();

    ZipArchive& operator=(ZipArchive&&) = default;
    ZipArchive& operator=(const ZipArchive&) = delete;

    void close();
    
    int file_count() const;

    std::vector<ZipArchiveFile> get_files() const;
    std::pair<bool, ZipArchiveFile> find_file(const std::filesystem::path& name) const;

    // flags:
    // 0 default zstd
    // 1 old libz
    bool add_file(const std::filesystem::path& filename, void* data, size_t size, int zip_flags);
    bool add_file(const std::filesystem::path& filename, ZipCustomSource* custom_source, int zip_flags);

    ZipIncrementalWrite begin_incremental_write(const std::filesystem::path& filename, int zip_flags);
private:

    int64_t add_file_internal(const std::filesystem::path& filename, zip_source* source, int zip_flags);

    zip* archive = nullptr;
};

}

#endif
