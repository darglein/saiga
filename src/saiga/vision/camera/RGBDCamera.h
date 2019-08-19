/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/vision/cameraModel/Distortion.h"
#include "saiga/vision/util/DepthmapPreprocessor.h"

#include <chrono>
#include <optional>

namespace Saiga
{
// The image types
using RGBImageType   = TemplatedImage<ucvec4>;
using DepthImageType = TemplatedImage<float>;

// All required intrinsics for the depth sensor
struct SAIGA_VISION_API RGBDIntrinsics
{
    // K matrix for depth and color
    // the image should already be registered
    StereoCamera4 K;
    Intrinsics4 depthK;
    Distortion dis;


    // Image options
    struct CameraOptions
    {
        int w = 640;
        int h = 480;
    };
    CameraOptions rgbo, deptho;

    int fps = 30;

    // Used to convert from the actual depth data to metric floats
    double depthFactor = 1.0;

    // Maximum depth (in meters) above which the depth values should be considered as outliers
    double maxDepth = 10;

    // The camera disconnects after this amount of frames
    int maxFrames = -1;
    // start frame for
    int startFrame = 0;



    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};


SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const RGBDIntrinsics& value);


struct SAIGA_VISION_API RGBDFrameData
{
    RGBImageType colorImg;
    DepthImageType depthImg;
    int frameId;
    std::chrono::steady_clock::time_point captureTime;

    // Some datasets provide ground truth pose estimations
    std::optional<SE3> groundTruth;
};

class SAIGA_VISION_API RGBDCamera
{
   public:
    RGBDCamera() {}
    RGBDCamera(const RGBDIntrinsics& intr) : _intrinsics(intr) {}
    virtual ~RGBDCamera() {}

    // Blocks until the next image is available
    virtual bool getImageSync(RGBDFrameData& data) = 0;

    // Returns false if no image is currently available
    virtual bool getImage(RGBDFrameData& data) { return getImageSync(data); }



    // Close the camera.
    // Blocking calls to waitForImage should return a 'nullptr'
    virtual void close() {}
    virtual bool isOpened() { return true; }


    const RGBDIntrinsics& intrinsics() { return _intrinsics; }

   protected:
    RGBDIntrinsics _intrinsics;
    int currentId = 0;

    // Create a frame data object with the images already allocated in the correct size
    void makeFrameData(RGBDFrameData& data);

    // Set frame id and capture time
    void setNextFrame(RGBDFrameData& data);
};

}  // namespace Saiga
