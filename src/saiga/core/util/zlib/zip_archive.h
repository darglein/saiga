
#pragma once

#include "saiga/config.h"
#include "saiga/core/util/ProgressBar.h"
#include <string>
#include <filesystem>

#if defined(SAIGA_USE_LIBZIP)

struct zip;

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

private:
    zip* archive = nullptr;
};

}

#endif
