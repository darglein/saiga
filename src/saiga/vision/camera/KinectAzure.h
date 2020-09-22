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
class SAIGA_VISION_API KinectCamera : public CameraBase
{
   public:
    struct KinectParams
    {
        // True: RGB, False: Greyscale
        bool color = true;

        // narrow or wide fov
        // Note: wide is only supported with 30 fps.
        bool narrow_depth = true;

        // Merge consecutive imu values
        int imu_merge_count = 4;

        int fps = 30;
    };


    KinectCamera(const KinectParams& kinect_params);
    ~KinectCamera();
    KinectCamera(const KinectCamera&) = delete;
    KinectCamera& operator=(const KinectCamera&) = delete;

    virtual bool getImageSync(FrameData& data) override;
    bool Open();


    std::string SerialNumber();
    const RGBDIntrinsics& intrinsics() { return _intrinsics; }



    Imu::Data GetImuSample(int num_merge_samples);



    k4a::calibration GetCalibration() { return calibration; }

   private:
    k4a_device_configuration_t config;
    k4a::device device;
    k4a::calibration calibration;
    RGBDIntrinsics _intrinsics;
    SE3 cam_to_imu;
    KinectParams kinect_params;
    k4a::transformation T;


    std::vector<Imu::Data> last_imu_data;
    double last_time = 0;
};

}  // namespace Saiga

#endif
