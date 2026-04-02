#include "libavif_wrapper.h"

#ifdef SAIGA_USE_LIBAVIF

#    include "EbSvtAv1Enc.h"  // The native SVT-AV1 header

#    include <avif/avif.h>
#    include <fstream>
#    include <iostream>

namespace Saiga
{
bool loadImageLibAVIF(const std::filesystem::path& path, Image& img)
{
    return false;
}

void SvtLogSilencer(void* context, SvtAv1LogLevel level, const char* tag, const char* fmt, va_list args)
{
    if (level <= SVT_AV1_LOG_ERROR)
    {
        std::cerr << "[SVT-AV1 ERROR] ";
        vfprintf(stderr, fmt, args);
    }
}

bool saveImageLibAVIF(const std::filesystem::path& path, const Image& img)
{
    if (img.type != ImageType::UC1 && img.type != ImageType::UC3 && img.type != ImageType::UC4 &&
        img.type != ImageType::US1 && img.type != ImageType::US3)
    {
        return false;
    }

    svt_av1_set_log_callback(SvtLogSilencer, nullptr);

    int channels = Saiga::channels(img.type);
    uint32_t width = img.width;
    uint32_t height = img.height;

    avifPixelFormat yuvFormat = AVIF_PIXEL_FORMAT_YUV420;

    // --- BIT DEPTH CHANGE POINT 1 ---
    // Change this to 10, 12, or 14 for higher bit depths.
    int bitDepth = elementSize(elementType(img.type)) * 8;
    int originalBitDepth = bitDepth;
    if (bitDepth == 16) bitDepth = 10;

    // 2. Create the main AVIF image container
    avifImage* image = avifImageCreate(width, height, bitDepth, yuvFormat);
    if (!image)
    {
        std::cerr << "Failed to create avifImage." << std::endl;
        return false;
    }

    // 3. Populate the image data based on the format
    if (channels == 1)
    {
        // GRAYSCALE: We skip RGB conversion and write directly to the Y plane.
        avifImageAllocatePlanes(image, AVIF_PLANES_YUV);

        // 1. Copy the Luminance (Y) data
        // Automatically handle the stride for 16-bit integers if bitDepth > 8
        size_t copyBytes = (bitDepth > 8) ? (width * 2) : width;

        for (int y = 0; y < height; ++y)
        {
            if (bitDepth == 8)
            {
                uint8_t* targetRow = image->yuvPlanes[AVIF_CHAN_Y] + (y * image->yuvRowBytes[AVIF_CHAN_Y]);
                const void* sourceRow = img.rowPtr(y);

                std::memcpy(targetRow, sourceRow, copyBytes);
            }
            else if (bitDepth == 10)
            {
                uint16_t* targetRow = reinterpret_cast<uint16_t*>(image->yuvPlanes[AVIF_CHAN_Y] + (y * image->yuvRowBytes[AVIF_CHAN_Y]));
                const uint16_t* sourceRow = reinterpret_cast<const uint16_t*>(img.rowPtr(y));

                for (int x = 0; x < width; ++x)
                {
                    // Shift 16-bit down to 10-bit (65535 -> 1023)
                    targetRow[x] = sourceRow[x] >> 6;
                }
            }
        }

        // --- NEW: PACIFY SVT-AV1 BY FAKING THE COLOR PLANES ---
        // For 4:2:0, the UV planes are exactly half the width and height of the Y plane.
        uint32_t uvHeight = (height + 1) / 2;
        uint32_t uvWidth = (width + 1) / 2;

        // The "neutral" color value depends on the bit depth.
        // 8-bit = 128 | 10-bit = 512 | 12-bit = 2048 | 14-bit = 8192
        int neutralChroma = 1 << (bitDepth - 1);

        for (uint32_t y = 0; y < uvHeight; ++y)
        {
            if (bitDepth == 8)
            {
                // For 8-bit, we can efficiently use memset
                uint8_t* uRow = image->yuvPlanes[AVIF_CHAN_U] + (y * image->yuvRowBytes[AVIF_CHAN_U]);
                uint8_t* vRow = image->yuvPlanes[AVIF_CHAN_V] + (y * image->yuvRowBytes[AVIF_CHAN_V]);
                memset(uRow, neutralChroma, uvWidth);
                memset(vRow, neutralChroma, uvWidth);
            }
            else
            {
                // For >8-bit, we cast the pointers to 16-bit and fill manually
                uint16_t* uRow = reinterpret_cast<uint16_t*>(image->yuvPlanes[AVIF_CHAN_U] + (y * image->yuvRowBytes[AVIF_CHAN_U]));
                uint16_t* vRow = reinterpret_cast<uint16_t*>(image->yuvPlanes[AVIF_CHAN_V] + (y * image->yuvRowBytes[AVIF_CHAN_V]));
                for (uint32_t x = 0; x < uvWidth; ++x)
                {
                    uRow[x] = neutralChroma;
                    vRow[x] = neutralChroma;
                }
            }
        }
    }
    else if (channels == 3 || channels == 4)
    {
        // RGB or RGBA: We use the helper struct to convert to YUV
        avifRGBImage rgbImage;
        avifRGBImageSetDefaults(&rgbImage, image);

        rgbImage.format = (channels == 3) ? AVIF_RGB_FORMAT_RGB : AVIF_RGB_FORMAT_RGBA;
        rgbImage.pixels = (uint8_t*)img.data8();
        rgbImage.rowBytes = img.pitchBytes;
        rgbImage.depth = originalBitDepth;

        // Convert RGB to YUV
        avifResult convertResult = avifImageRGBToYUV(image, &rgbImage);
        if (convertResult != AVIF_RESULT_OK)
        {
            std::cerr << "Failed to convert RGB to YUV: " << avifResultToString(convertResult) << std::endl;
            avifImageDestroy(image);
            return false;
        }
    }
    else
    {
        std::cerr << "Unsupported number of channels: " << channels << std::endl;
        avifImageDestroy(image);
        return false;
    }

    // 4. Set up the Encoder
    avifEncoder* encoder = avifEncoderCreate();
    // Quality ranges from 0 (lossless) to 100 (worst). 60 is a good default for lossy.
    encoder->quality      = img.get_compression_quality();
    encoder->qualityAlpha = img.get_compression_quality();
    encoder->speed        = AVIF_SPEED_DEFAULT;

    // 5. Encode the Image
    avifRWData output       = AVIF_DATA_EMPTY;
    avifResult encodeResult = avifEncoderWrite(encoder, image, &output);

    if (encodeResult != AVIF_RESULT_OK)
    {
        std::cerr << "Encoding failed: " << avifResultToString(encodeResult) << std::endl;
        avifEncoderDestroy(encoder);
        avifImageDestroy(image);
        return false;
    }

    // 6. Save to disk
    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        avifRWDataFree(&output);
        avifEncoderDestroy(encoder);
        avifImageDestroy(image);
        return false;
    }

    file.write(reinterpret_cast<const char*>(output.data), output.size);
    file.close();

    // 7. Cleanup
    avifRWDataFree(&output);
    avifEncoderDestroy(encoder);
    avifImageDestroy(image);

    return true;
}

bool loadImageFromMemoryLibAVIF(const void* data, size_t size, Image& img)
{
    return false;
}
}  // namespace Saiga

#endif
