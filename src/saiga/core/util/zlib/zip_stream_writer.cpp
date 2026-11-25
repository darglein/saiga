/**
 * Copyright (c) 2026 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "zip_stream_writer.h"

#ifdef SAIGA_USE_ZLIB

#    include <ctime>
#    include <stdexcept>
#    include <zlib.h>

namespace Saiga
{
namespace
{
// Field values from the zip specification (PKWARE APPNOTE 6.3.x).
constexpr uint32_t kLocalHeaderSignature   = 0x04034b50;
constexpr uint32_t kDescriptorSignature    = 0x08074b50;
constexpr uint32_t kCentralHeaderSignature = 0x02014b50;
constexpr uint32_t kZip64EndSignature      = 0x06064b50;
constexpr uint32_t kZip64LocatorSignature  = 0x07064b50;
constexpr uint32_t kEndSignature           = 0x06054b50;

constexpr uint16_t kFlagDataDescriptor = 1 << 3;
constexpr uint16_t kFlagUtf8Name       = 1 << 11;
constexpr uint16_t kZip64ExtraId       = 0x0001;
constexpr uint16_t kVersionStored      = 10;
constexpr uint16_t kVersionZip64       = 45;

constexpr uint32_t kMax32 = 0xFFFFFFFFu;
constexpr uint16_t kMax16 = 0xFFFFu;

class Bytes
{
   public:
    void u16(uint16_t v)
    {
        for (int i = 0; i < 2; ++i) data.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
    void u32(uint32_t v)
    {
        for (int i = 0; i < 4; ++i) data.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
    void u64(uint64_t v)
    {
        for (int i = 0; i < 8; ++i) data.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
    void str(const std::string& s) { data.insert(data.end(), s.begin(), s.end()); }

    std::vector<uint8_t> data;
};

uint32_t clamp32(uint64_t v)
{
    return v >= kMax32 ? kMax32 : static_cast<uint32_t>(v);
}

void current_dos_time(uint16_t& dos_time, uint16_t& dos_date)
{
    const std::time_t now = std::time(nullptr);
    std::tm t{};
#    ifdef _WIN32
    localtime_s(&t, &now);
#    else
    localtime_r(&now, &t);
#    endif
    if (t.tm_year < 80)
    {
        // DOS dates start in 1980.
        dos_time = 0;
        dos_date = (1 << 5) | 1;
        return;
    }
    dos_time = static_cast<uint16_t>((t.tm_hour << 11) | (t.tm_min << 5) | (t.tm_sec / 2));
    dos_date = static_cast<uint16_t>(((t.tm_year - 80) << 9) | ((t.tm_mon + 1) << 5) | t.tm_mday);
}

uint32_t update_crc(uint32_t crc, const void* data, size_t size)
{
    // crc32_z returns 0, not `crc`, for a null buffer.
    if (size == 0) return crc;
    return static_cast<uint32_t>(crc32_z(crc, static_cast<const Bytef*>(data), size));
}

}  // namespace

ZipFileSink::ZipFileSink(const std::filesystem::path& path) : file_(path, std::ios::binary | std::ios::trunc)
{
    if (!file_.is_open())
    {
        const auto name = path.u8string();  // std::u8string since C++20
        throw std::runtime_error("ZipFileSink: cannot create " + std::string(name.begin(), name.end()));
    }
}

void ZipFileSink::write(const void* data, size_t size)
{
    file_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!file_)
    {
        throw std::runtime_error("ZipFileSink: write failed");
    }
}

void ZipFileSink::flush()
{
    file_.flush();
    if (!file_)
    {
        throw std::runtime_error("ZipFileSink: flush failed");
    }
}

ZipStreamWriter::ZipStreamWriter(ZipStreamSink* sink) : sink_(sink)
{
    if (!sink_)
    {
        throw std::logic_error("ZipStreamWriter: null sink");
    }
    current_dos_time(dos_time_, dos_date_);
}

ZipStreamWriter::ZipStreamWriter(const std::filesystem::path& path) : file_(std::in_place, path), sink_(&*file_)
{
    current_dos_time(dos_time_, dos_date_);
}

ZipStreamWriter::~ZipStreamWriter() = default;

uint64_t ZipStreamWriter::add_entry(const std::string& name, const void* data, size_t size)
{
    check_writable("add_entry");
    if (in_entry_)
    {
        throw std::logic_error("ZipStreamWriter: add_entry while an entry is open");
    }
    const uint64_t data_offset = start_entry(name, false, size, update_crc(0, data, size));
    emit(data, size);
    return data_offset;
}

uint64_t ZipStreamWriter::begin_entry(const std::string& name)
{
    check_writable("begin_entry");
    if (in_entry_)
    {
        throw std::logic_error("ZipStreamWriter: begin_entry while an entry is open");
    }
    const uint64_t data_offset = start_entry(name, true, 0, 0);
    in_entry_                  = true;
    return data_offset;
}

void ZipStreamWriter::write(const void* data, size_t size)
{
    check_writable("write");
    if (!in_entry_)
    {
        throw std::logic_error("ZipStreamWriter: write without an open entry");
    }
    Entry& entry = entries_.back();
    entry.crc    = update_crc(entry.crc, data, size);
    entry.size += size;
    emit(data, size);
}

void ZipStreamWriter::end_entry()
{
    check_writable("end_entry");
    if (!in_entry_)
    {
        throw std::logic_error("ZipStreamWriter: end_entry without an open entry");
    }
    // The local header announced ZIP64, so the descriptor carries 64-bit sizes.
    const Entry& entry = entries_.back();
    Bytes d;
    d.u32(kDescriptorSignature);
    d.u32(entry.crc);
    d.u64(entry.size);
    d.u64(entry.size);
    emit(d.data.data(), d.data.size());
    in_entry_ = false;
}

void ZipStreamWriter::finish()
{
    check_writable("finish");
    if (in_entry_)
    {
        throw std::logic_error("ZipStreamWriter: finish while an entry is open");
    }

    const uint64_t cd_offset = offset_;
    for (const Entry& entry : entries_)
    {
        Bytes extra;
        if (entry.size >= kMax32)
        {
            // Uncompressed and compressed size, which are the same for stored entries.
            extra.u64(entry.size);
            extra.u64(entry.size);
        }
        if (entry.header_offset >= kMax32)
        {
            extra.u64(entry.header_offset);
        }
        const bool zip64 = !extra.data.empty() || entry.zip64_local_header;

        Bytes h;
        h.u32(kCentralHeaderSignature);
        h.u16(kVersionZip64);  // version made by, MS-DOS attributes
        h.u16(zip64 ? kVersionZip64 : kVersionStored);
        h.u16(entry.flags);
        h.u16(0);  // stored
        h.u16(dos_time_);
        h.u16(dos_date_);
        h.u32(entry.crc);
        h.u32(clamp32(entry.size));
        h.u32(clamp32(entry.size));
        h.u16(static_cast<uint16_t>(entry.name.size()));
        h.u16(static_cast<uint16_t>(extra.data.empty() ? 0 : 4 + extra.data.size()));
        h.u16(0);  // comment length
        h.u16(0);  // disk number
        h.u16(0);  // internal attributes
        h.u32(0);  // external attributes
        h.u32(clamp32(entry.header_offset));
        h.str(entry.name);
        if (!extra.data.empty())
        {
            h.u16(kZip64ExtraId);
            h.u16(static_cast<uint16_t>(extra.data.size()));
            h.data.insert(h.data.end(), extra.data.begin(), extra.data.end());
        }
        emit(h.data.data(), h.data.size());
    }
    const uint64_t cd_size = offset_ - cd_offset;
    const uint64_t count   = entries_.size();

    Bytes end;
    if (count >= kMax16 || cd_offset >= kMax32 || cd_size >= kMax32)
    {
        const uint64_t zip64_end_offset = offset_;
        end.u32(kZip64EndSignature);
        end.u64(44);  // size of the rest of this record
        end.u16(kVersionZip64);
        end.u16(kVersionZip64);
        end.u32(0);  // this disk
        end.u32(0);  // disk with the central directory
        end.u64(count);
        end.u64(count);
        end.u64(cd_size);
        end.u64(cd_offset);

        end.u32(kZip64LocatorSignature);
        end.u32(0);  // disk with the ZIP64 end record
        end.u64(zip64_end_offset);
        end.u32(1);  // total disks
    }
    end.u32(kEndSignature);
    end.u16(0);
    end.u16(0);
    end.u16(static_cast<uint16_t>(count >= kMax16 ? kMax16 : count));
    end.u16(static_cast<uint16_t>(count >= kMax16 ? kMax16 : count));
    end.u32(clamp32(cd_size));
    end.u32(clamp32(cd_offset));
    end.u16(0);  // comment length
    emit(end.data.data(), end.data.size());

    finished_ = true;
    flush();
}

void ZipStreamWriter::flush()
{
    if (!file_)
    {
        return;
    }
    try
    {
        file_->flush();
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

uint64_t ZipStreamWriter::start_entry(const std::string& name, bool streamed, uint64_t size, uint32_t crc)
{
    if (name.empty() || name.size() > kMax16)
    {
        throw std::logic_error("ZipStreamWriter: entry name must have 1 to 65535 bytes");
    }

    Entry entry;
    entry.name          = name;
    entry.flags         = kFlagUtf8Name | (streamed ? kFlagDataDescriptor : 0);
    entry.crc           = crc;
    entry.size          = size;
    entry.header_offset = offset_;
    // A streamed entry may grow past 4 GiB, and its descriptor can only have 64-bit sizes if the local header has
    // a ZIP64 field. With unknown sizes that field holds zeros, as the data descriptor has the real values.
    entry.zip64_local_header = streamed || size >= kMax32;

    Bytes h;
    h.u32(kLocalHeaderSignature);
    h.u16(entry.zip64_local_header ? kVersionZip64 : kVersionStored);
    h.u16(entry.flags);
    h.u16(0);  // stored
    h.u16(dos_time_);
    h.u16(dos_date_);
    h.u32(crc);
    h.u32(entry.zip64_local_header ? kMax32 : static_cast<uint32_t>(size));
    h.u32(entry.zip64_local_header ? kMax32 : static_cast<uint32_t>(size));
    h.u16(static_cast<uint16_t>(name.size()));
    h.u16(entry.zip64_local_header ? 20 : 0);
    h.str(name);
    if (entry.zip64_local_header)
    {
        h.u16(kZip64ExtraId);
        h.u16(16);
        h.u64(size);
        h.u64(size);
    }
    emit(h.data.data(), h.data.size());
    entries_.push_back(std::move(entry));
    return offset_;
}

void ZipStreamWriter::emit(const void* data, size_t size)
{
    if (size == 0)
    {
        return;
    }
    try
    {
        sink_->write(data, size);
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
    offset_ += size;
}

void ZipStreamWriter::check_writable(const char* call) const
{
    if (failed_)
    {
        throw std::runtime_error(std::string("ZipStreamWriter: ") + call + " after a failed write");
    }
    if (finished_)
    {
        throw std::logic_error(std::string("ZipStreamWriter: ") + call + " after finish");
    }
}

}  // namespace Saiga

#endif
