#pragma once

#include "saiga/config.h"
#include "saiga/core/util/ProgressBar.h"
#include <string>
#include <filesystem>
// #include <zip.h>

#if defined(SAIGA_USE_LIBZIP)

struct zip;
struct zip_source;
struct zip_file;
typedef struct zip_file zip_file_t;

namespace Saiga
{

enum class ZipMode
{
    Read,
    Write,
};

enum class ZipCompressionMethod
{
    UNCOMPRESSED,
    ZSTD,
    ZLIB,
};

struct SAIGA_CORE_API ZipArchiveFile
{
    std::filesystem::path filename;
    size_t compressed_size;
    size_t uncompressed_size;
    zip* archive;



    bool read_all(void* out_data, ProgressBarManager* progress_bar = nullptr);


    bool open();
    bool close();
    bool read(void* out_data, size_t size) ;

private:
    zip_file_t* file = 0;
};

struct SAIGA_CORE_API ZipIncrementalWrite
{
    ZipIncrementalWrite() = default;
    ZipIncrementalWrite(zip* archive, zip_source* source) : archive(archive), source(source) {}
    ZipIncrementalWrite(ZipIncrementalWrite&& o) noexcept;
    ~ZipIncrementalWrite();

    void operator=(ZipIncrementalWrite&& o) noexcept;

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
    ZipArchive(ZipArchive&& o) noexcept;
    ZipArchive(const ZipArchive&) = delete;
    ~ZipArchive();

    ZipArchive& operator=(ZipArchive&&) = default;
    ZipArchive& operator=(const ZipArchive&) = delete;

    void close();
    
    int file_count() const;

    std::vector<ZipArchiveFile> get_files() const;
    std::pair<bool, ZipArchiveFile> find_file(const std::filesystem::path& name) const;

    bool add_file(const std::filesystem::path& filename, void* data, size_t size, ZipCompressionMethod method);
    bool add_file(const std::filesystem::path& filename, ZipCustomSource* custom_source, ZipCompressionMethod method);

    ZipIncrementalWrite begin_incremental_write(const std::filesystem::path& filename, ZipCompressionMethod method);
private:

    int64_t add_file_internal(const std::filesystem::path& filename, zip_source* source, ZipCompressionMethod method);

    zip* archive = nullptr;
};

}

#endif
