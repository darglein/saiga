#include "zip_archive.h"

#if defined(SAIGA_USE_LIBZIP)

#    include <iostream>
#    include <utility>
#    include <zip.h>

namespace Saiga
{

static void print_error(const char* prefix, int err)
{
    zip_error_t error;
    zip_error_init_with_code(&error, err);
    std::cout << prefix << "\n" << zip_error_strerror(&error);
    zip_error_fini(&error);
}

ZipArchive::ZipArchive(const std::filesystem::path& path, ZipMode mode)
{
    if(mode == ZipMode::Write)
    {
        std::filesystem::remove(path);
    }
    int flag = mode == ZipMode::Read ? ZIP_RDONLY : (ZIP_CREATE | ZIP_TRUNCATE);

    int error = 0;
    archive   = zip_open(path.u8string().c_str(), flag, &error);
    if (!archive)
    {
        print_error("ZIP: Failed to open or create archive.", error);
    }
}

ZipArchive::ZipArchive(ZipArchive&& o) noexcept : archive(std::exchange(o.archive, nullptr)) {}

ZipArchive::~ZipArchive()
{
    close();
}

void ZipArchive::close()
{
    if (archive)
    {
        if (zip_close(archive) != 0)
        {
            std::cout << "ZIP: Failed to write ZIP archive Zip Error: " << zip_strerror(archive) << std::endl;
        }
        archive = nullptr;
    }
}

int ZipArchive::file_count() const
{
    if (!archive)
    {
        return 0;
    }
    // return zip_get_num_files(archive);
    return zip_get_num_entries(archive, 0);
}

static ZipArchiveFile file_from_stat(zip_stat_t stat, zip* archive)
{
    ZipArchiveFile file;
    file.filename          = std::filesystem::u8path(stat.name);
    file.compressed_size   = stat.comp_size;
    file.uncompressed_size = stat.size;
    file.archive           = archive;
    return file;
}

std::vector<ZipArchiveFile> ZipArchive::get_files() const
{
    std::vector<ZipArchiveFile> result;

    if (archive)
    {
        // int numFiles = zip_get_num_files(archive);
        int numFiles = zip_get_num_entries(archive, 0);
        for (int i = 0; i < numFiles; ++i)
        {
            zip_stat_t stat;
            if (zip_stat_index(archive, i, ZIP_FL_ENC_UTF_8, &stat) == 0)
            {
                result.push_back(file_from_stat(stat, archive));
            }
        }
    }

    return result;
}

std::pair<bool, ZipArchiveFile> ZipArchive::find_file(const std::filesystem::path& name) const
{
    if (archive)
    {
        zip_stat_t stat;
        if (zip_stat(archive, name.u8string().c_str(), ZIP_FL_ENC_UTF_8, &stat) == 0)
        {
            return {true, file_from_stat(stat, archive)};
        }
    }

    return {false, {}};
}

bool ZipArchive::add_file(const std::filesystem::path& filename, void* data, size_t size, ZipCompressionMethod method)
{
    if (!archive)
    {
        return false;
    }

    zip_source* source = zip_source_buffer(archive, data, size, 0);
    if (!source)
    {
        std::cout << "ZIP: Failed to create source buffer.\n" << zip_strerror(archive);
        return false;
    }

    auto index = add_file_internal(filename, source, method);
    if (index < 0)
    {
        std::cout << "ZIP: Failed to add file to archive: " << zip_strerror(archive) << '\n';
        zip_source_free(source);
        return false;
    }

    return true;
}

ZipIncrementalWrite ZipArchive::begin_incremental_write(const std::filesystem::path& filename, ZipCompressionMethod method)
{
    if (!archive)
    {
        return {};
    }

    zip_source* source = zip_source_buffer_create(nullptr, 0, 0, nullptr);
    if (!source)
    {
        std::cout << "ZIP: Failed to create source buffer.\n" << zip_strerror(archive);
        return {};
    }

    auto index = add_file_internal(filename, source, method);
    if (index < 0)
    {
        std::cout << "ZIP: Failed to add file to archive: " << zip_strerror(archive) << '\n';
        zip_source_free(source);
        return {};
    }

    if (zip_source_begin_write(source) < 0) 
    {
        std::cout << "ZIP: Failed to begin write: " << zip_strerror(archive) << '\n';
        zip_source_free(source);
        return {};
    }

    return ZipIncrementalWrite(archive, source);
}

bool ZipIncrementalWrite::write(void* data, size_t size)
{
    if (zip_source_write(source, data, size) < 0)
    {
        std::cout << "ZIP: Failed to write block: " << zip_strerror(archive) << '\n';
        zip_source_rollback_write(source);
        return false;
    }

    return true;
}

ZipIncrementalWrite::ZipIncrementalWrite(ZipIncrementalWrite&& o) noexcept
    : archive(std::exchange(o.archive, nullptr)), source(std::exchange(o.source, nullptr))
{
}

ZipIncrementalWrite::~ZipIncrementalWrite()
{
    if (source)
    {
        if (zip_source_commit_write(source) < 0) 
        {
            std::cout << "ZIP: Failed to commit write: " << zip_strerror(archive) << '\n';
            zip_source_rollback_write(source);
        }
    }
}




static zip_int64_t source_callback(void* userdata, void* data, zip_uint64_t len, zip_source_cmd_t cmd)
{
    ZipCustomSource* source = (ZipCustomSource*)userdata;

    switch (cmd) 
    {
        case ZIP_SOURCE_READ: 
        {
            return source->read_next(data, len);
        }

        case ZIP_SOURCE_STAT: 
        {
            zip_stat_t* st = (zip_stat_t*)data;
            zip_stat_init(st);

            st->valid = ZIP_STAT_SIZE;
            st->size = source->total_size();

            return sizeof(*st);
        }

        case ZIP_SOURCE_OPEN:
        case ZIP_SOURCE_CLOSE:
        case ZIP_SOURCE_FREE:
            return 0;

        default:
            return -1;
    }
}


bool ZipArchive::add_file(const std::filesystem::path& filename, ZipCustomSource* custom_source, ZipCompressionMethod method)
{
    if (!archive)
    {
        return {};
    }

    zip_source_t* source = zip_source_function(archive, source_callback, custom_source);
    if (!source)
    {
        std::cout << "ZIP: Failed to create callback source.\n";
        return 1;
    }

    auto index = add_file_internal(filename, source, method);
    if (index < 0)
    {
        std::cout << "ZIP: Failed to add file to archive: " << zip_strerror(archive) << '\n';
        zip_source_free(source);
        return false;
    }

    return true;
}

int64_t ZipArchive::add_file_internal(const std::filesystem::path& filename, zip_source* source, ZipCompressionMethod method)
{
    auto index = (int)zip_file_add(archive, filename.u8string().c_str(), source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);

    auto libzip_method = ZIP_CM_ZSTD;
    switch (method)
    {
        case ZipCompressionMethod::UNCOMPRESSED:
            libzip_method = ZIP_CM_STORE;
            break;
        case ZipCompressionMethod::ZSTD:
            libzip_method = ZIP_CM_ZSTD;
            break;
        case ZipCompressionMethod::ZLIB:
            libzip_method = ZIP_CM_DEFAULT;
            break;
    }

    zip_set_file_compression(archive, index, libzip_method, 0);

    return index;
}


bool ZipArchiveFile::open()
{
    file = zip_fopen(archive, filename.u8string().c_str(), ZIP_FL_ENC_UTF_8);
    return true;
}

bool ZipArchiveFile::close()
{
    zip_fclose(file);
    file = 0;
    return true;
}

bool ZipArchiveFile::read(void* out_data, size_t size)
{
    auto read_result = zip_fread(file, out_data, size);
    SAIGA_ASSERT(size ==  read_result);
    return true;
}

bool ZipArchiveFile::read_all(void* out_data, ProgressBarManager* progress_bar)
{
    if (!archive)
    {
        return false;
    }

    open();
    if (!file)
    {
        std::cout << "ZIP: Failed to open file in archive.\n";
        return false;
    }


    int read_block_size = 1024;

    size_t num_blocks = (uncompressed_size + read_block_size - 1) / read_block_size;
    auto bar          = SAIGA_OPTIONAL_PROGRESS_BAR(progress_bar, "Unzip", num_blocks);

    uint8_t* byte_ptr = (uint8_t*)out_data;
    size_t cursor     = 0;

    zip_int64_t read_result = 0;
    do
    {
        read_result = zip_fread(file, byte_ptr + cursor, read_block_size);
        if (read_result < 0)
        {
            std::cout << "ZIP: Failed to read from file: " << zip_file_strerror(file) << '\n';
            return false;
        }

        cursor += read_result;

        if (bar) bar->addProgress(1);
    } while (read_result > 0);


    zip_fread(file, out_data, uncompressed_size);
    close();
    return true;
}

}  // namespace Saiga
#endif
