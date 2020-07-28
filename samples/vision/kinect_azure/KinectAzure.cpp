/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */
#include "KinectAzure.h"

#include "saiga/core/image/all.h"
#include "saiga/core/util/assert.h"

#include <iomanip>
#include <iostream>

#ifdef SAIGA_USE_K4A
namespace Saiga
{
KinectCamera::KinectCamera()
{
    Open();


    // Intrinsics from internal calibration
    {
        auto color              = calibration.color_camera_calibration;
        _intrinsics.imageSize.w = color.resolution_width;
        _intrinsics.imageSize.h = color.resolution_height;

        auto params = color.intrinsics.parameters.param;

        _intrinsics.model.K.fx = params.fx;
        _intrinsics.model.K.fy = params.fy;
        _intrinsics.model.K.cx = params.cx;
        _intrinsics.model.K.cy = params.cy;

        _intrinsics.model.dis.k1 = params.k1;
        _intrinsics.model.dis.k2 = params.k2;
        _intrinsics.model.dis.k3 = params.k3;
        _intrinsics.model.dis.k4 = params.k4;
        _intrinsics.model.dis.k5 = params.k5;
        _intrinsics.model.dis.k6 = params.k6;

        _intrinsics.model.dis.p1 = params.p1;
        _intrinsics.model.dis.p2 = params.p2;
    }

    {
        //        auto depth                   = calibration.depth_camera_calibration;
        //        _intrinsics.depthImageSize.w = depth.resolution_width;
        //        _intrinsics.depthImageSize.h = depth.resolution_height;
        _intrinsics.depthImageSize = _intrinsics.imageSize;
    }

    {
        _intrinsics.bgr = true;
    }

    //    std::cout << "params " << color.intrinsics.parameter_count << std::endl;
    //    auto intr = color.intrinsics.parameters;
}

KinectCamera::~KinectCamera()
{
    if (device)
    {
        device.close();
    }
}


bool KinectCamera::Open()
{
    SAIGA_ASSERT(!device);
    uint32_t device_count = k4a_device_get_installed_count();

    if (device_count == 0)
    {
        std::cout << "No K4A devices found" << std::endl;
        return false;
    }

    device = k4a::device::open(K4A_DEVICE_DEFAULT);
    SAIGA_ASSERT(device);


    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.color_format               = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution           = K4A_COLOR_RESOLUTION_720P;
    config.depth_mode                 = K4A_DEPTH_MODE_NFOV_UNBINNED;
    //    config.depth_mode               = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    config.camera_fps               = K4A_FRAMES_PER_SECOND_30;
    config.synchronized_images_only = true;

    _intrinsics.fps = 15;

    device.start_cameras(&config);
    calibration = device.get_calibration(config.depth_mode, config.color_resolution);


    std::cout << "Kinect Azure Opened. ID: " << SerialNumber() << std::endl;
    std::cout << "   Color: " << calibration.color_camera_calibration.resolution_width << " x "
              << calibration.color_camera_calibration.resolution_height << std::endl;
    std::cout << "   Depth: " << calibration.depth_camera_calibration.resolution_width << " x "
              << calibration.depth_camera_calibration.resolution_height << std::endl;

    return true;
}

bool KinectCamera::getImageSync(RGBDFrameData& data)
{
    data.colorImg.create(intrinsics().imageSize.h, intrinsics().imageSize.w);
    data.depthImg.create(intrinsics().depthImageSize.h, intrinsics().depthImageSize.w);
    data.id = currentId++;

    k4a::capture capture;


    if (!device.get_capture(&capture, std::chrono::milliseconds(1000)))
    {
        std::cout << "Capture Timeout" << std::endl;
        return false;
    }

    // Probe for a color image
    k4a::image image;
    image = capture.get_color_image();
    if (image)
    {
        ImageView<ucvec4> view(image.get_height_pixels(), image.get_width_pixels(), image.get_stride_bytes(),
                               image.get_buffer());
        view.copyTo(data.colorImg.getImageView());
    }

    // probe for a IR16 image
    image = capture.get_ir_image();
    if (image)
    {
    }
    else
    {
    }

    // Probe for a depth16 image
    image = capture.get_depth_image();
    if (image)
    {
        static k4a::transformation T(calibration);
        SAIGA_BLOCK_TIMER();
        auto transformed_depth = T.depth_image_to_color_camera(image);


        ImageView<unsigned short> view2(image.get_height_pixels(), image.get_width_pixels(), image.get_stride_bytes(),
                                        image.get_buffer());


        ImageView<unsigned short> view(transformed_depth.get_height_pixels(), transformed_depth.get_width_pixels(),
                                       transformed_depth.get_stride_bytes(), transformed_depth.get_buffer());


#    if 1
        //        int i = 400;
        //        int j = 300;
        for (auto i : view.rowRange())
        {
            for (auto j : view.colRange())
            {
                auto d              = view(i, j);
                data.depthImg(i, j) = d / 1000.f;
            }
        }
#    endif
    }


    return true;
}

std::string KinectCamera::SerialNumber()
{
    return device.get_serialnum();
}


void print_calibration()
{
    using std::cout;
    using std::endl;

    uint32_t device_count = k4a_device_get_installed_count();
    cout << "Found " << device_count << " connected devices:" << endl;
    cout << std::fixed << std::setprecision(6);

    for (uint8_t deviceIndex = 0; deviceIndex < device_count; deviceIndex++)
    {
        k4a_device_t device = NULL;
        if (K4A_RESULT_SUCCEEDED != k4a_device_open(deviceIndex, &device))
        {
            cout << deviceIndex << ": Failed to open device" << endl;
            exit(-1);
        }

        k4a_calibration_t calibration;

        k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        deviceConfig.color_format               = K4A_IMAGE_FORMAT_COLOR_MJPG;
        deviceConfig.color_resolution           = K4A_COLOR_RESOLUTION_1080P;
        deviceConfig.depth_mode                 = K4A_DEPTH_MODE_NFOV_UNBINNED;
        deviceConfig.camera_fps                 = K4A_FRAMES_PER_SECOND_30;
        deviceConfig.wired_sync_mode            = K4A_WIRED_SYNC_MODE_STANDALONE;
        deviceConfig.synchronized_images_only   = true;

        // get calibration
        if (K4A_RESULT_SUCCEEDED !=
            k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &calibration))
        {
            cout << "Failed to get calibration" << endl;
            exit(-1);
        }

        auto calib = calibration.depth_camera_calibration;

        cout << "resolution width: " << calib.resolution_width << endl;
        cout << "resolution height: " << calib.resolution_height << endl;
        cout << "principal point x: " << calib.intrinsics.parameters.param.cx << endl;
        cout << "principal point y: " << calib.intrinsics.parameters.param.cy << endl;
        cout << "focal length x: " << calib.intrinsics.parameters.param.fx << endl;
        cout << "focal length y: " << calib.intrinsics.parameters.param.fy << endl;
        cout << "radial distortion coefficients:" << endl;
        cout << "k1: " << calib.intrinsics.parameters.param.k1 << endl;
        cout << "k2: " << calib.intrinsics.parameters.param.k2 << endl;
        cout << "k3: " << calib.intrinsics.parameters.param.k3 << endl;
        cout << "k4: " << calib.intrinsics.parameters.param.k4 << endl;
        cout << "k5: " << calib.intrinsics.parameters.param.k5 << endl;
        cout << "k6: " << calib.intrinsics.parameters.param.k6 << endl;
        cout << "center of distortion in Z=1 plane, x: " << calib.intrinsics.parameters.param.codx << endl;
        cout << "center of distortion in Z=1 plane, y: " << calib.intrinsics.parameters.param.cody << endl;
        cout << "tangential distortion coefficient x: " << calib.intrinsics.parameters.param.p1 << endl;
        cout << "tangential distortion coefficient y: " << calib.intrinsics.parameters.param.p2 << endl;
        cout << "metric radius: " << calib.intrinsics.parameters.param.metric_radius << endl;

        k4a_device_close(device);
    }
}



}  // namespace Saiga
#endif
