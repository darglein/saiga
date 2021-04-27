/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/cameraModel/CameraModel.h"
#include "saiga/vision/imu/Imu.h"

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
    Mono    = 0,
    RGBD    = 1,
    Stereo  = 2,
    Unknown = 3,
};


/**
 * The base class for all different camera types.
 */
struct SAIGA_VISION_API FrameMetaData
{
    // The first frame should have id = 0
    // -1 means the frame is invalid.
    int id = -1;

    // Capture time in seconds. Should always be interpreted relative.
    // -1 means no timestamp was recorded.
    double timeStamp = -1;

    // Some datasets provide ground truth poses
    std::optional<SE3> groundTruth;

    Imu::ImuSequence imu_data;

    void Save(const std::string& dir) const;
    void Load(const std::string& dir);
};

/**
 * Data from a monocular camera.
 * In some cases a gray image instead of rgb is transmitted.
 * Use colorImg.valid() to check for a good image.
 */
struct SAIGA_VISION_API FrameData : public FrameMetaData
{
    // The image data either as rgb or gray image.
    // In stereo mode, this is the left camera image.
    GrayImageType image;
    RGBImageType image_rgb;
    std::string image_file;

    // Only valid in RGBD mode.
    DepthImageType depth_image;
    std::string depth_file;

    // Only valid in stereo mode.
    // Contains the right camera image.
    GrayImageType right_image;
    RGBImageType right_image_rgb;
    std::string right_image_file;

    CameraInputType CameraType();

    void Save(const std::string& dir) const;
    void Load(const std::string& dir);

    void FreeImageData()
    {
        image.free();
        image_rgb.free();
        depth_image.free();
        right_image.free();
        right_image_rgb.free();
    }
};


struct SAIGA_VISION_API MonocularIntrinsics
{
    // All color images are stored in BGR instead of RGB
    bool bgr = false;

    int fps = -1;
    ImageDimensions imageSize;
    PinholeCamera model;


    // Transforms the model-pose of the camera to the mode-pose of the body by right multiplication.
    //
    //  SE3 camera_pose = camera_view.inverse();
    //  SE3 body_pose = camera_pose * camera_to_body;
    SE3 camera_to_body;

    SE3 camera_to_gt;

    StereoCamera4 dummyStereoCamera() const { return StereoCamera4(model.K, 1); }

    static constexpr CameraInputType cameraType = CameraInputType::Mono;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const MonocularIntrinsics& value);



struct SAIGA_VISION_API StereoIntrinsics : public MonocularIntrinsics
{
    ImageDimensions rightImageSize;
    PinholeCamera rightModel;

    // BaseLine * fx
    double bf = 0;

    // Maximum depth (in meters) above which the depth values should be considered as outliers
    double maxDepth = 10;


    SE3 left_to_right;

    StereoCamera4 stereoCamera() const
    {
        SAIGA_ASSERT(bf != 0);
        return StereoCamera4(model.K, bf);
    }
};


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

    StereoCamera4 stereoCamera() const
    {
        SAIGA_ASSERT(bf != 0);
        return StereoCamera4(model.K, bf);
    }
    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);


    static constexpr CameraInputType cameraType = CameraInputType::RGBD;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const RGBDIntrinsics& value);


SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const StereoIntrinsics& value);

}  // namespace Saiga
