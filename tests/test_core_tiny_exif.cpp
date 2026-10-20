/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Separate from test_core_util.cpp: exif.h defines the PARSE_EXIF_* error codes as macros, which clash with the
// enum of the same names in TinyEXIF.h.
#include "saiga/core/util/exif/TinyEXIF.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace TinyEXIF;

namespace
{
// Builds a little endian EXIF segment ("Exif\0\0" + TIFF header). Offsets of the entries are relative to the TIFF
// header, which starts at byte 6.
struct ExifBuilder
{
    std::vector<uint8_t> data;

    ExifBuilder()
    {
        const char header[] = "Exif\0\0II";
        data.insert(data.end(), header, header + 8);
        U16(0x2a);
        U32(8);
    }
    void U16(uint16_t v)
    {
        data.push_back(v & 0xFF);
        data.push_back(v >> 8);
    }
    void U32(uint32_t v)
    {
        U16(v & 0xFFFF);
        U16(v >> 16);
    }
    void Entry(uint16_t tag, uint16_t format, uint32_t length, uint32_t value)
    {
        U16(tag);
        U16(format);
        U32(length);
        U32(value);
    }
    // Offset of the next byte relative to the TIFF header
    uint32_t Offset() const { return uint32_t(data.size() - 6); }
};

// Wraps an EXIF segment into a minimal JPEG: SOI, APP1, EOI.
std::vector<uint8_t> MakeJpeg(const std::vector<uint8_t>& exif)
{
    std::vector<uint8_t> jpeg = {0xFF, 0xD8, 0xFF, 0xE1};
    size_t length             = exif.size() + 2;
    jpeg.push_back(uint8_t(length >> 8));
    jpeg.push_back(uint8_t(length & 0xFF));
    jpeg.insert(jpeg.end(), exif.begin(), exif.end());
    jpeg.push_back(0xFF);
    jpeg.push_back(0xD9);
    return jpeg;
}

// IFD0 with the camera make "Saiga" stored after the IFD.
ExifBuilder MakeExifWithMake()
{
    ExifBuilder b;
    uint32_t make_offset = 8 + 2 + 12 + 4;
    b.U16(1);
    b.Entry(0x10F, 2, 6, make_offset);
    b.U32(0);
    const char make[] = "Saiga";
    b.data.insert(b.data.end(), make, make + 6);
    return b;
}
}  // namespace

TEST(TinyEXIF, ParseJpeg)
{
    auto jpeg = MakeJpeg(MakeExifWithMake().data);
    EXIFInfo info;
    ASSERT_EQ(info.parseFrom(jpeg.data(), unsigned(jpeg.size())), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info.Make, "Saiga");
}

TEST(TinyEXIF, EmptyString)
{
    // A string entry with zero values read the character at index -1 of the empty string.
    ExifBuilder b;
    b.U16(1);
    b.Entry(0x10F, 2, 0, 0);
    b.U32(0);

    EXIFInfo info;
    ASSERT_EQ(info.parseFromEXIFSegment(b.data.data(), unsigned(b.data.size())), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info.Make, "");
}

TEST(TinyEXIF, RationalOutsideOfSegment)
{
    // Rationals were read from their offset without checking it against the segment length.
    ExifBuilder b;
    uint32_t value = 8 + 2 + 12 + 4;
    b.U16(1);
    b.Entry(0x011a, 5, 1, value);  // XResolution
    b.U32(0);
    ASSERT_EQ(b.Offset(), value);
    size_t segment_end = b.data.size();
    // A valid rational behind the end of the segment that is passed to the parser.
    b.U32(72);
    b.U32(1);

    EXIFInfo info;
    ASSERT_EQ(info.parseFromEXIFSegment(b.data.data(), unsigned(b.data.size())), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info.XResolution, 72);

    EXIFInfo info2;
    ASSERT_EQ(info2.parseFromEXIFSegment(b.data.data(), unsigned(segment_end)), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info2.XResolution, 0);
}

TEST(TinyEXIF, GPSOutsideOfSegment)
{
    // The GPS rationals were read without checking the offset against the segment length.
    ExifBuilder b;
    uint32_t gps_ifd = 8 + 2 + 12 + 4;
    b.U16(1);
    b.Entry(0x8825, 4, 1, gps_ifd);
    b.U32(0);
    ASSERT_EQ(b.Offset(), gps_ifd);
    uint32_t lat = gps_ifd + 2 + 2 * 12 + 4;
    b.U16(2);
    b.Entry(2, 5, 3, lat);  // latitude
    b.Entry(6, 5, 1, lat);  // altitude
    b.U32(0);
    ASSERT_EQ(b.Offset(), lat);
    size_t segment_end = b.data.size();
    for (int i = 0; i < 3; ++i)
    {
        b.U32(10);
        b.U32(1);
    }

    EXIFInfo info;
    ASSERT_EQ(info.parseFromEXIFSegment(b.data.data(), unsigned(b.data.size())), PARSE_EXIF_SUCCESS);
    EXPECT_NEAR(info.GeoLocation.Latitude, 10 + 10 / 60.0 + 10 / 3600.0, 1e-9);
    EXPECT_EQ(info.GeoLocation.Altitude, 10);

    EXIFInfo info2;
    ASSERT_EQ(info2.parseFromEXIFSegment(b.data.data(), unsigned(segment_end)), PARSE_EXIF_SUCCESS);
    // Not set
    EXPECT_EQ(info2.GeoLocation.Latitude, std::numeric_limits<double>::max());
    EXPECT_FALSE(info2.GeoLocation.hasAltitude());
}

TEST(TinyEXIF, GPSTimeStampOutsideOfSegment)
{
    // The timestamp was stored as "0 0 0" (midnight) when its rationals were outside of the segment.
    ExifBuilder b;
    uint32_t gps_ifd = 8 + 2 + 12 + 4;
    b.U16(1);
    b.Entry(0x8825, 4, 1, gps_ifd);
    b.U32(0);
    ASSERT_EQ(b.Offset(), gps_ifd);
    uint32_t time = gps_ifd + 2 + 12 + 4;
    b.U16(1);
    b.Entry(7, 5, 3, time);
    b.U32(0);
    ASSERT_EQ(b.Offset(), time);
    size_t segment_end = b.data.size();
    for (uint32_t v : {12, 30, 15})
    {
        b.U32(v);
        b.U32(1);
    }

    EXIFInfo info;
    ASSERT_EQ(info.parseFromEXIFSegment(b.data.data(), unsigned(b.data.size())), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info.GeoLocation.GPSTimeStamp, "12 30 15");

    EXIFInfo info2;
    ASSERT_EQ(info2.parseFromEXIFSegment(b.data.data(), unsigned(segment_end)), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info2.GeoLocation.GPSTimeStamp, "");
}

TEST(TinyEXIF, SegmentLongerThanBuffer)
{
    // The length of the APP1 segment was not checked against the buffer, so the EXIF data behind the end of the buffer
    // was parsed.
    auto jpeg = MakeJpeg(MakeExifWithMake().data);
    EXIFInfo info;
    EXPECT_NE(info.parseFrom(jpeg.data(), unsigned(jpeg.size() - 10)), PARSE_EXIF_SUCCESS);
    EXPECT_EQ(info.Make, "");

    // A segment length below 2 underflowed.
    std::vector<uint8_t> broken = {0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x01};
    broken.resize(32, 0);
    EXPECT_NE(info.parseFrom(broken.data(), unsigned(broken.size())), PARSE_EXIF_SUCCESS);

    // A marker at the end of the buffer without length bytes.
    std::vector<uint8_t> truncated(16, 0);
    truncated[0]  = 0xFF;
    truncated[1]  = 0xD8;
    truncated[14] = 0xFF;
    truncated[15] = 0xE2;
    EXPECT_NE(info.parseFrom(truncated.data(), unsigned(truncated.size())), PARSE_EXIF_SUCCESS);
}
