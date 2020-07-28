/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once


#include "saiga/config.h"
#include "saiga/vision/camera/CameraBase.h"

#ifdef SAIGA_USE_K4A
#    include "saiga/core/image/image.h"
#    include "saiga/core/util/Thread/SynchronizedBuffer.h"
#    include "saiga/vision/camera/CameraBase.h"

#    include <k4a/k4a.hpp>
#    include <string>
#    include <thread>
// Helper functions to access the Kinect Azure.
// Most of these are copied from samples in the Azure SDK.
namespace Saiga
{
class KinectCamera : public CameraBase<RGBDFrameData>
{
   public:
    KinectCamera();
    ~KinectCamera();
    KinectCamera(const KinectCamera&) = delete;
    KinectCamera& operator=(const KinectCamera&) = delete;

    virtual bool getImageSync(RGBDFrameData& data);
    bool Open();


    std::string SerialNumber();
    const RGBDIntrinsics& intrinsics() { return _intrinsics; }

   private:
    k4a::device device;
    k4a::calibration calibration;
    RGBDIntrinsics _intrinsics;
};


void print_calibration();
}  // namespace Saiga

#endif
