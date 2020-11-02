/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"
//
#include "saiga/core/image/imageView.h"

#ifdef SAIGA_USE_CUDA_TOOLKIT

#    include <npp.h>
#    include <nppi.h>

#    define CHECK_NPPI_ERROR(function)                                                                              \
        {                                                                                                           \
            auto error_code = function;                                                                             \
            ((error_code == NPP_SUCCESS)                                                                            \
                 ? static_cast<void>(0)                                                                             \
                 : Saiga::saiga_assert_fail(#function " == NPP_SUCCESS", __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION, \
                                            "Error code: " + std::to_string(error_code)));                          \
        }

#    if (NPP_VERSION_MAJOR > 10 || (NPP_VERSION_MAJOR == 10 && NPP_VERSION_MINOR >= 1)) && \
        !defined(NPPI_USE_OLD_CONTEXT)
#        define SAIGA_NPPI_HAS_STREAM_CONTEXT
#    endif

#    ifndef SAIGA_NPPI_HAS_STREAM_CONTEXT
struct SaigaNppStreamContext
{
    cudaStream_t hStream;
};
#    else
using SaigaNppStreamContext = NppStreamContext;
#    endif

// This file contains helper functions for interfacing with the NPPI library.
// As NPPI mostly consists of image processing utilities, most method have
// ImageViews as parameters. This is also a reference, if you want to apply
// similar operations on images of different types. Then you usually can just
// copy/paste the function and rename the nppi call.
namespace Saiga
{
namespace NPPI
{
inline SaigaNppStreamContext CreateStreamContextWithStream(cudaStream_t stream)
{
    SaigaNppStreamContext context;
#    ifdef SAIGA_NPPI_HAS_STREAM_CONTEXT
    CHECK_NPPI_ERROR(nppGetStreamContext(&context));
#    endif
    context.hStream = stream;
    return context;
}

template <typename T>
inline NppiSize GetSize(ImageView<T> img)
{
    NppiSize size;
    size.width  = img.width;
    size.height = img.height;
    return size;
}

template <typename T>
inline NppiRect GetRoi(ImageView<T> img)
{
    NppiRect size;
    size.x      = 0;
    size.y      = 0;
    size.width  = img.width;
    size.height = img.height;
    return size;
}

inline void GaussFilter(ImageView<const unsigned char> src, ImageView<unsigned char> dst,
                        const SaigaNppStreamContext& context)
{
    NppiPoint offset;
    offset.x = 0;
    offset.y = 0;
#    ifdef SAIGA_NPPI_HAS_STREAM_CONTEXT
    CHECK_NPPI_ERROR(nppiFilterGaussBorder_8u_C1R_Ctx(src.data8, src.pitchBytes, GetSize(src), offset, dst.data8,
                                                      dst.pitchBytes, GetSize(dst), NPP_MASK_SIZE_7_X_7,
                                                      NPP_BORDER_REPLICATE, context));
#    else
    CHECK_NPPI_ERROR(nppiFilterGaussBorder_8u_C1R(src.data8, src.pitchBytes, GetSize(src), offset, dst.data8,
                                                  dst.pitchBytes, GetSize(dst), NPP_MASK_SIZE_7_X_7,
                                                  NPP_BORDER_REPLICATE));
#    endif
}

inline void ResizeLinear(ImageView<const unsigned char> src, ImageView<unsigned char> dst,
                         const SaigaNppStreamContext& context)
{
#    ifdef SAIGA_NPPI_HAS_STREAM_CONTEXT
    CHECK_NPPI_ERROR(nppiResize_8u_C1R_Ctx(src.data8, src.pitchBytes, GetSize(src), GetRoi(src), dst.data8,
                                           dst.pitchBytes, GetSize(dst), GetRoi(dst), NPPI_INTER_LINEAR, context));
#    else
    CHECK_NPPI_ERROR(nppiResize_8u_C1R(src.data8, src.pitchBytes, GetSize(src), GetRoi(src), dst.data8, dst.pitchBytes,
                                       GetSize(dst), GetRoi(dst), NPPI_INTER_LINEAR));
#    endif
}

}  // namespace NPPI
}  // namespace Saiga
#endif
