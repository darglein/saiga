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
KinectCamera::KinectCamera(const KinectParams& kinect_params) : kinect_params(kinect_params)
{
    camera_type = CameraInputType::RGBD;
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
        _intrinsics.depthModel     = _intrinsics.model;
    }

    {
        _intrinsics.bgr = true;

        // Approx. 10cm Baseline
        _intrinsics.bf = 0.1 * _intrinsics.model.K.fx;
    }


    auto get_extrinsics = [&](auto src, auto dst) {
        auto extr = calibration.extrinsics[src][dst];

        Mat3 R = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(extr.rotation).cast<double>();

        // t is given in (mm)
        Vec3 t = Eigen::Map<Eigen::Matrix<float, 3, 1>>(extr.translation).cast<double>() * (1.0 / 1000.0);

        SE3 T(R, t);
        return T;
    };


    auto cam_to_gyro = get_extrinsics(K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_GYRO);
    auto cam_to_acc  = get_extrinsics(K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_ACCEL);

    cam_to_imu = SE3(cam_to_gyro.so3(), cam_to_acc.translation()).inverse();

    _intrinsics.camera_to_body = cam_to_imu;

    imu                 = Imu::Sensor();
    imu->frequency      = 1666 / kinect_params.imu_merge_count;
    imu->frequency_sqrt = sqrt(imu->frequency);

    std::cout << _intrinsics << std::endl;



    for (int i = 0; i < 10; ++i)
    {
        FrameData d;
        getImageSync(d);
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


    config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;

    if (kinect_params.color)
    {
        config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    }
    else
    {
        config.color_format = K4A_IMAGE_FORMAT_COLOR_NV12;
    }

    if (kinect_params.narrow_depth)
    {
        config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    }
    else
    {
        config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
    }

    config.color_resolution = K4A_COLOR_RESOLUTION_720P;


    if (kinect_params.fps == 15)
    {
        config.camera_fps = K4A_FRAMES_PER_SECOND_15;
        _intrinsics.fps   = 15;
    }
    else
    {
        config.camera_fps = K4A_FRAMES_PER_SECOND_30;
        _intrinsics.fps   = 30;
    }

    config.synchronized_images_only = true;


    device.start_cameras(&config);
    device.start_imu();
    calibration = device.get_calibration(config.depth_mode, config.color_resolution);

    T = k4a::transformation(calibration);

    std::cout << "Kinect Azure Opened. ID: " << SerialNumber() << std::endl;
    std::cout << "   Color: " << calibration.color_camera_calibration.resolution_width << " x "
              << calibration.color_camera_calibration.resolution_height << std::endl;
    std::cout << "   Depth: " << calibration.depth_camera_calibration.resolution_width << " x "
              << calibration.depth_camera_calibration.resolution_height << std::endl;



    return true;
}

bool KinectCamera::getImageSync(FrameData& data)
{
    data.id = currentId++;

    k4a::capture capture;


    if (!device.get_capture(&capture, std::chrono::milliseconds(1000)))
    {
        std::cout << "Capture Timeout" << std::endl;
        return false;
    }


    k4a::image k4a_color = capture.get_color_image();
    k4a::image k4a_depth = capture.get_depth_image();

    double t_color = k4a_color.get_device_timestamp().count() / (1000.0 * 1000.0);
    //    double t_depth = k4a_depth.get_device_timestamp().count() / (1000.0 * 1000.0);

    data.timeStamp = t_color;


    std::vector<Imu::Data> imu_data;
    while (true)
    {
        Imu::Data sample = GetImuSample(8);
        imu_data.push_back(sample);

        //        std::cout << sample << std::endl;
        if (sample.timestamp >= t_color)
        {
            break;
        }
    }
    //    std::cout << "imu data size: " << imu_data.size() << " Imu frequency: " << imu_data.size() * 30 << std::endl;


    // Add last 2 samples at the front
    if (last_imu_data.size() >= 2)
    {
        imu_data.insert(imu_data.begin(), last_imu_data.end() - 2, last_imu_data.end());
        data.imu_data.data       = imu_data;
        data.imu_data.time_begin = last_time;
        data.imu_data.time_end   = data.timeStamp;

        data.imu_data.FixBorder();
        SAIGA_ASSERT(data.imu_data.Valid());
        SAIGA_ASSERT(data.imu_data.complete());
    }

    last_imu_data = imu_data;
    last_time     = data.timeStamp;

    SAIGA_ASSERT(k4a_color);


    if (config.color_format == K4A_IMAGE_FORMAT_COLOR_BGRA32)
    {
        // Probe for a color image
        data.image_rgb.create(intrinsics().imageSize.h, intrinsics().imageSize.w);

        ImageView<ucvec4> view(k4a_color.get_height_pixels(), k4a_color.get_width_pixels(),
                               k4a_color.get_stride_bytes(), k4a_color.get_buffer());
        view.copyTo(data.image_rgb.getImageView());

        data.image_rgb.getImageView().swapChannels(0, 2);
    }
    else if (config.color_format == K4A_IMAGE_FORMAT_COLOR_NV12)
    {
        ImageView<unsigned char> view(k4a_color.get_height_pixels(), k4a_color.get_width_pixels(),
                                      k4a_color.get_stride_bytes(), k4a_color.get_buffer());
        data.image.create(intrinsics().imageSize.h, intrinsics().imageSize.w);

        view.copyTo(data.image.getImageView());
    }
    else
    {
        SAIGA_EXIT_ERROR("unknown color format");
    }



    if (k4a_depth)
    {
        data.depth_image.create(intrinsics().depthImageSize.h, intrinsics().depthImageSize.w);

        //        SAIGA_BLOCK_TIMER();
        auto transformed_depth = T.depth_image_to_color_camera(k4a_depth);


        ImageView<unsigned short> view2(k4a_depth.get_height_pixels(), k4a_depth.get_width_pixels(),
                                        k4a_depth.get_stride_bytes(), k4a_depth.get_buffer());


        ImageView<unsigned short> view(transformed_depth.get_height_pixels(), transformed_depth.get_width_pixels(),
                                       transformed_depth.get_stride_bytes(), transformed_depth.get_buffer());

        for (auto i : view.rowRange())
        {
            for (auto j : view.colRange())
            {
                auto d                 = view(i, j);
                data.depth_image(i, j) = d / 1000.f;
            }
        }
    }



    // Access the accelerometer readings
    //    if (imu_sample != NULL)
    //    {
    //        printf(" | Accelerometer temperature:%.2f x:%.4f y:%.4f z: %.4f\n", imu_sample.temperature,
    //               imu_sample.acc_sample.xyz.x, imu_sample.acc_sample.xyz.y, imu_sample.acc_sample.xyz.z);
    //    }

    return true;
}

std::string KinectCamera::SerialNumber()
{
    return device.get_serialnum();
}

Imu::Data KinectCamera::GetImuSample(int num_merge_samples)
{
    Imu::Data result;
    result.omega.setZero();
    result.acceleration.setZero();
    result.timestamp = 0;

    for (int i = 0; i < num_merge_samples; ++i)
    {
        k4a_imu_sample_t imu_sample;
        // if (device.get_imu_sample(&imu_sample, std::chrono::milliseconds(0)))
        if (device.get_imu_sample(&imu_sample, std::chrono::milliseconds(K4A_WAIT_INFINITE)))
        {
            double t   = imu_sample.gyro_timestamp_usec / (1000.0 * 1000);
            Vec3 acc   = Vec3(imu_sample.acc_sample.xyz.x, imu_sample.acc_sample.xyz.y, imu_sample.acc_sample.xyz.z);
            Vec3 omega = Vec3(imu_sample.gyro_sample.xyz.x, imu_sample.gyro_sample.xyz.y, imu_sample.gyro_sample.xyz.z);

            result.timestamp += t;
            result.acceleration += acc;
            result.omega += omega;
        }
        else
        {
            SAIGA_EXIT_ERROR("Could not read imu sample");
        }
    }

    double f = 1.0 / num_merge_samples;
    result.timestamp *= f;
    result.acceleration *= f;
    result.omega *= f;
    return result;

    // Combine all samples into a single sample
}



}  // namespace Saiga
#endif
