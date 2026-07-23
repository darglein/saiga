#include "libavif_wrapper.h"

#ifdef SAIGA_USE_LIBAVIF

#    if defined(SAIGA_USE_SVTAV1)
#        include "EbSvtAv1Enc.h"  // Native SVT-AV1 header for log callback / init
#    endif

#    include <avif/avif.h>
#    include <fstream>
#    include <iostream>
#    include <mutex>

namespace Saiga
{
bool loadImageLibAVIF(const std::filesystem::path& path, Image& img)
{
    return false;
}

#    if defined(SAIGA_USE_SVTAV1)
static std::once_flag svt_init_flag;
#    endif

bool saveImageLibAVIF(const std::filesystem::path& path, const Image& img)
{
    if (img.type != ImageType::UC1 && img.type != ImageType::UC3 && img.type != ImageType::UC4 &&
        img.type != ImageType::US1 && img.type != ImageType::US3)
    {
        return false;
    }


    auto encoder_choice = AVIF_CODEC_CHOICE_AUTO;

#    if defined(SAIGA_USE_LIBAOM)
    encoder_choice = AVIF_CODEC_CHOICE_AOM;
#    elif defined(SAIGA_USE_SVTAV1)
    encoder_choice = AVIF_CODEC_CHOICE_SVT;
#else
#error No AVIF codec available
#    endif


#    if defined(SAIGA_USE_SVTAV1)
    if (encoder_choice == AVIF_CODEC_CHOICE_SVT)
    {
        // SVT-AV1 specific thread-safety & logger warm-up workaround.
        std::call_once(svt_init_flag,
                       []()
                       {
                           auto svt_log_silencer =
                               [](void* context, SvtAv1LogLevel level, const char* tag, const char* fmt, va_list args)
                           {
                               if (level <= SVT_AV1_LOG_ERROR)
                               {
                                   std::cerr << "[SVT-AV1 ERROR] ";
                                   vfprintf(stderr, fmt, args);
                               }
                           };

                           svt_av1_set_log_callback(svt_log_silencer, nullptr);

                           // Perform a tiny dummy encode to force SVT-AV1 to run its internal
                           // RTCD (Run-Time CPU Detect) SIMD initialization sequentially.
                           avifEncoder* dummy_encoder = avifEncoderCreate();

                           // Force the dummy encoder to specifically trigger SVT-AV1
                           dummy_encoder->codecChoice = AVIF_CODEC_CHOICE_SVT;

                           avifImage* dummy_image = avifImageCreate(4, 4, 8, AVIF_PIXEL_FORMAT_YUV420);
                           avifImageAllocatePlanes(dummy_image, AVIF_PLANES_YUV);
                           avifRWData dummy_output = AVIF_DATA_EMPTY;

                           avifEncoderWrite(dummy_encoder, dummy_image, &dummy_output);

                           avifRWDataFree(&dummy_output);
                           avifImageDestroy(dummy_image);
                           avifEncoderDestroy(dummy_encoder);
                       });
    }
#    endif

    int channels    = Saiga::channels(img.type);
    uint32_t width  = img.width;
    uint32_t height = img.height;

    avifPixelFormat yuvFormat = AVIF_PIXEL_FORMAT_YUV420;

    if (encoder_choice == AVIF_CODEC_CHOICE_AOM)
    {
        yuvFormat = (channels == 1) ? AVIF_PIXEL_FORMAT_YUV400 : AVIF_PIXEL_FORMAT_YUV444;
    }

    int bitDepth         = elementSize(elementType(img.type)) * 8;
    int originalBitDepth = bitDepth;
    if (bitDepth == 16) bitDepth = 10;

    // 1. Create the main AVIF image container
    avifImage* image = avifImageCreate(width, height, bitDepth, yuvFormat);
    if (!image)
    {
        std::cerr << "Failed to create avifImage." << std::endl;
        return false;
    }

    // 2. Populate image data
    if (channels == 1)
    {
        // GRAYSCALE: Skip RGB conversion and write directly to the Y plane.
        avifImageAllocatePlanes(image, AVIF_PLANES_YUV);

        size_t copyBytes = (bitDepth > 8) ? (width * 2) : width;

        for (int y = 0; y < height; ++y)
        {
            if (bitDepth == 8)
            {
                uint8_t* targetRow    = image->yuvPlanes[AVIF_CHAN_Y] + (y * image->yuvRowBytes[AVIF_CHAN_Y]);
                const void* sourceRow = img.rowPtr(y);
                std::memcpy(targetRow, sourceRow, copyBytes);
            }
            else if (bitDepth == 10)
            {
                uint16_t* targetRow =
                    reinterpret_cast<uint16_t*>(image->yuvPlanes[AVIF_CHAN_Y] + (y * image->yuvRowBytes[AVIF_CHAN_Y]));
                const uint16_t* sourceRow = reinterpret_cast<const uint16_t*>(img.rowPtr(y));

                for (int x = 0; x < width; ++x)
                {
                    targetRow[x] = sourceRow[x] >> 6;
                }
            }
        }

        if (yuvFormat != AVIF_PIXEL_FORMAT_YUV400)
        {
            SAIGA_ASSERT(yuvFormat == AVIF_PIXEL_FORMAT_YUV420);

            uint32_t uvHeight = (height + 1) / 2;
            uint32_t uvWidth = (width + 1) / 2;
            int neutralChroma = 1 << (bitDepth - 1);

            for (uint32_t y = 0; y < uvHeight; ++y)
            {
                if (bitDepth == 8)
                {
                    uint8_t* uRow = image->yuvPlanes[AVIF_CHAN_U] + (y * image->yuvRowBytes[AVIF_CHAN_U]);
                    uint8_t* vRow = image->yuvPlanes[AVIF_CHAN_V] + (y * image->yuvRowBytes[AVIF_CHAN_V]);
                    memset(uRow, neutralChroma, uvWidth);
                    memset(vRow, neutralChroma, uvWidth);
                }
                else
                {
                    uint16_t* uRow =
                        reinterpret_cast<uint16_t*>(image->yuvPlanes[AVIF_CHAN_U] + (y * image->yuvRowBytes[AVIF_CHAN_U]));
                    uint16_t* vRow =
                        reinterpret_cast<uint16_t*>(image->yuvPlanes[AVIF_CHAN_V] + (y * image->yuvRowBytes[AVIF_CHAN_V]));
                    for (uint32_t x = 0; x < uvWidth; ++x)
                    {
                        uRow[x] = neutralChroma;
                        vRow[x] = neutralChroma;
                    }
                }
            }
        }
    }
    else if (channels == 3 || channels == 4)
    {
        avifRGBImage rgbImage;
        avifRGBImageSetDefaults(&rgbImage, image);

        rgbImage.format   = (channels == 3) ? AVIF_RGB_FORMAT_RGB : AVIF_RGB_FORMAT_RGBA;
        rgbImage.pixels   = (uint8_t*)img.data8();
        rgbImage.rowBytes = img.pitchBytes;
        rgbImage.depth    = originalBitDepth;

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

    // 3. Set up the Encoder
    avifEncoder* encoder  = avifEncoderCreate();
    encoder->codecChoice  = encoder_choice;
    encoder->quality      = img.get_compression_quality();
    encoder->qualityAlpha = img.get_compression_quality();
    encoder->speed        = AVIF_SPEED_FASTEST;
    encoder->maxThreads   = 1;

    // 4. Encode the Image
    avifRWData output       = AVIF_DATA_EMPTY;
    avifResult encodeResult = avifEncoderWrite(encoder, image, &output);

    if (encodeResult != AVIF_RESULT_OK)
    {
        std::cerr << "Encoding failed: " << avifResultToString(encodeResult) << std::endl;
        avifEncoderDestroy(encoder);
        avifImageDestroy(image);
        return false;
    }

    // 5. Save to disk
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

    // 6. Cleanup
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
