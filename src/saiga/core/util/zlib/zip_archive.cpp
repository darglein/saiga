#include "zip_archive.h"

#if defined(SAIGA_USE_LIBZIP)

#    include <iostream>
#    include <utility>
#    include <zip.h>

namespace Saiga
{

ZipArchive::ZipArchive(const std::filesystem::path& path, ZipMode mode)
{
    int flag = mode == ZipMode::Read ? ZIP_RDONLY : (ZIP_CREATE | ZIP_TRUNCATE);

    int error = 0;
    archive   = zip_open(path.string().c_str(), flag, &error);
    if (!archive)
    {
        std::cout << "ZIP: Failed to open or create archive.\n";
    }
}

ZipArchive::ZipArchive(ZipArchive&& o) : archive(std::exchange(o.archive, nullptr)) {}

ZipArchive::~ZipArchive()
{
    close();
}

void ZipArchive::close()
{
    if (archive)
    {
        zip_close(archive);
        archive = nullptr;
    }
}

int ZipArchive::file_count() const
{
    if (!archive)
    {
        return 0;
    }
    return zip_get_num_files(archive);
}

static ZipArchiveFile file_from_stat(zip_stat_t stat, zip* archive)
{
    ZipArchiveFile file;
    file.filename          = stat.name;
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
        int numFiles = zip_get_num_files(archive);
        for (int i = 0; i < numFiles; ++i)
        {
            zip_stat_t stat;
            if (zip_stat_index(archive, i, 0, &stat) == 0)
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
        if (zip_stat(archive, name.string().c_str(), 0, &stat) == 0)
        {
            return {true, file_from_stat(stat, archive)};
        }
    }

    return {false, {}};
}

bool ZipArchive::add_file(const std::filesystem::path& filename, void* data, size_t size, int zip_flags)
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

    auto index = (int)zip_file_add(archive, filename.string().c_str(), source, ZIP_FL_OVERWRITE);

    // ZIP_CM_STORE uncompressed
    // ZIP_CM_DEFAULT
    // ZIP_CM_ZSTD
    if (zip_flags == 0)
    {
        zip_set_file_compression(archive, index, ZIP_CM_ZSTD, 0);
    }
    else
    {
        zip_set_file_compression(archive, index, ZIP_CM_DEFAULT, 0);
    }

    if (index < 0)
    {
        std::cout << "ZIP: Failed to add file to archive: " << zip_strerror(archive) << '\n';
        zip_source_free(source);
        return false;
    }

    return true;
}

bool ZipArchiveFile::read(void* out_data, ProgressBarManager* progress_bar) const
{
    if (!archive)
    {
        return false;
    }

    zip_file_t* file = zip_fopen(archive, filename.string().c_str(), 0);
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
    zip_fclose(file);
    return true;
}

}  // namespace Saiga
#endif
