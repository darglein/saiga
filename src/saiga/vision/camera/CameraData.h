/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/cameraModel/CameraModel.h"

#include <optional>

namespace Saiga
{
// Image types used for different vision operations.
using RGBImageType       = TemplatedImage<ucvec4>;
using RGBAImageType      = TemplatedImage<ucvec4>;
using DepthImageType     = TemplatedImage<float>;
using GrayImageType      = TemplatedImage<unsigned char>;
using GrayFloatImageType = TemplatedImage<float>;

enum class CameraInputType : int
{
    Mono   = 0,
    RGBD   = 1,
    Stereo = 2,
};


/**
 * The base class for all different camera types.
 */
struct FrameMetaData
{
    // The first frame should have id = 0
    // -1 means the frame is invalid.
    int id = -1;

    // Capture time in seconds. Should always be interpreted relative.
    // -1 means no timestamp was recorded.
    double timeStamp = -1;

    // Some datasets provide ground truth poses
    std::optional<SE3> groundTruth;
};

/**
 * Data from a monocular camera.
 * In some cases a gray image instead of rgb is transmitted.
 * Use colorImg.valid() to check for a good image.
 */
struct MonocularFrameData : public FrameMetaData
{
    GrayImageType grayImg;
    RGBImageType colorImg;

    static constexpr CameraInputType cameraType = CameraInputType::Mono;
};

/**
 * Color + Depth
 */
struct RGBDFrameData : public MonocularFrameData
{
    DepthImageType depthImg;

    static constexpr CameraInputType cameraType = CameraInputType::RGBD;
};

/**
 * Monocular + a second image
 */
struct StereoFrameData : public MonocularFrameData
{
    GrayImageType grayImg2;
    RGBImageType colorImg2;

    static constexpr CameraInputType cameraType = CameraInputType::Stereo;
};

struct BaseIntrinsics
{
    int fps = 30;
};

struct SAIGA_VISION_API MonocularIntrinsics : public BaseIntrinsics
{
    ImageDimensions imageSize;
    PinholeCamera model;


    static constexpr CameraInputType cameraType = CameraInputType::Mono;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const MonocularIntrinsics& value);

// All required intrinsics for the depth sensor
struct SAIGA_VISION_API RGBDIntrinsics : public MonocularIntrinsics
{
    ImageDimensions depthImageSize;
    PinholeCamera depthModel;

    // BaseLine * fx
    double bf = 0;

    // Used to convert from the actual depth data to metric floats
    double depthFactor = 1.0;

    // Maximum depth (in meters) above which the depth values should be considered as outliers
    double maxDepth = 10;

    StereoCamera4 stereoCamera() const { return StereoCamera4(model.K, bf); }
    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);


    static constexpr CameraInputType cameraType = CameraInputType::RGBD;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const RGBDIntrinsics& value);

struct SAIGA_VISION_API StereoIntrinsics : public MonocularIntrinsics
{
    ImageDimensions rightImageSize;
    PinholeCamera rightModel;

    // BaseLine * fx
    double bf = 0;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const StereoIntrinsics& value);

}  // namespace Saiga
