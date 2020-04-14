/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/time/timer.h"
#include "saiga/vision/imu/Imu.h"

#include "CameraData.h"

#include <fstream>
#include <iomanip>
#include <thread>

namespace Saiga
{
class CameraBase2
{
   public:
    virtual ~CameraBase2() {}

    virtual void close() {}
    virtual bool isOpened() { return true; }
    // transforms the pose to the ground truth reference frame.
    virtual SE3 CameraToGroundTruth() { return SE3(); }
    SE3 GroundTruthToCamera() { return CameraToGroundTruth().inverse(); }


    // Optional IMU data if the camera provides it.
    // The returned vector contains all data from frame-1 to frame.
    virtual Imu::Frame ImuDataForFrame(int frame) { return {}; }
    virtual std::optional<Imu::Sensor> getIMU() { return {}; }
};

/**
 * Interface class for different datset inputs.
 */
template <typename _FrameType>
class SAIGA_TEMPLATE CameraBase : public CameraBase2
{
   public:
    using FrameType = _FrameType;
    virtual ~CameraBase() {}

    // Blocks until the next image is available
    // Returns true if success.
    virtual bool getImageSync(FrameType& data) = 0;

    // Returns false if no image is currently available
    virtual bool getImage(FrameType& data) { return getImageSync(data); }


   protected:
    int currentId = 0;
};


struct SAIGA_VISION_API DatasetParameters
{
    // The playback fps. Doesn't have to match the actual camera fps.
    double playback_fps = 30;

    // Root directory of the dataset. The exact value depends on the dataset type.
    std::string dir;

    // Ground truth file. Only used for the kitti dataset. The other datasets have them included in the main directory.
    std::string groundTruth;

    // Throw away all frames before 'startFrame'
    int startFrame = 0;

    // Only load 'maxFrames' after the startFrame.
    int maxFrames = -1;

    // Load images in parallel with omp
    bool multiThreadedLoad = true;


    // Load only the first image of the sequence. All other meta data (GT+IMU) are still loaded normally.
    // This can be used to save disk usage, if, for example, the features are precomputed and stored in a file.
    bool only_first_image = false;

    // Force monocular load. Don't load second image or depth map for stereo and rgbd sensors.
    bool force_monocular = false;

    // Load all images to ram at the beginning.
    bool preload = true;

    void fromConfigFile(const std::string& file);
};

/**
 * Interface for cameras that read datasets.
 */
template <typename FrameType>
class SAIGA_TEMPLATE DatasetCameraBase : public CameraBase<FrameType>
{
   public:
    DatasetCameraBase(const DatasetParameters& params) : params(params)
    {
        timeStep = std::chrono::duration_cast<tick_t>(
            std::chrono::duration<double, std::micro>(1000000.0 / params.playback_fps));
        timer.start();
        lastFrameTime = timer.stop();
        nextFrameTime = lastFrameTime + timeStep;
    }

    bool getImageSync(FrameType& data) override
    {
        if (!this->isOpened())
        {
            return false;
        }


        auto t = timer.stop();

        if (t < nextFrameTime)
        {
            std::this_thread::sleep_for(nextFrameTime - t);
            nextFrameTime += timeStep;
        }
        else if (t < nextFrameTime + timeStep)
        {
            nextFrameTime += timeStep;
        }
        else
        {
            nextFrameTime = t + timeStep;
        }


        auto& img = frames[this->currentId];
        SAIGA_ASSERT(this->currentId == img.id);
        if (!params.only_first_image || this->currentId == 0)
        {
            LoadImageData(img);
        }
        this->currentId++;
        data = std::move(img);
        return true;
    }

    virtual bool isOpened() override { return this->currentId < (int)frames.size(); }
    size_t getFrameCount() { return frames.size(); }


    virtual void LoadImageData(FrameType& data) {}

    // Saves the groundtruth in TUM-Trajectory format:
    // <timestamp> <translation x y z> <rotation x y z w>
    void saveGroundTruthTrajectory(const std::string& file)
    {
        std::ofstream strm(file);
        strm << std::setprecision(20);
        for (auto& f : frames)
        {
            double time = f.timeStamp;
            SAIGA_ASSERT(f.groundTruth);

            SE3 pose = f.groundTruth.value();
            Vec3 t   = pose.translation();
            Quat q   = pose.unit_quaternion();
            strm << time << " " << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z()
                 << " " << q.w() << std::endl;
        }
    }

    // Completely removes the frames between from and to
    void eraseFrames(int from, int to)
    {
        frames.erase(frames.begin() + from, frames.begin() + to);
        imuDataForFrame.erase(imuDataForFrame.begin() + from, imuDataForFrame.begin() + to);
    }

    void computeImuDataPerFrame()
    {
        // Create IMU per frame vector by adding all imu datas from frame_i to frame_i+1 to frame_i+1.
        imuDataForFrame.resize(frames.size());
        int currentImuid = 0;

        for (int i = 0; i < frames.size(); ++i)
        {
            Imu::Frame& imuFrame = imuDataForFrame[i];
            auto& a              = frames[i];
            imuFrame.timestamp   = a.timeStamp;
            for (; currentImuid < imuData.size(); ++currentImuid)
            {
                auto id = imuData[currentImuid];
                if (id.timestamp < a.timeStamp)
                {
                    imuFrame.imu_data_since_last_frame.push_back(id);
                }
                else
                {
                    imuFrame.imu_directly_after_this_frame = id;
                    break;
                }
            }

            // not a valid meassurement after this frame
            // -> use last one if it exists
            // This is not really correct but an approximation that only effects the last frame of a dataset.
            if (!std::isfinite(imuFrame.imu_directly_after_this_frame.timestamp))
            {
                if (!imuFrame.imu_data_since_last_frame.empty())
                {
                    imuFrame.imu_directly_after_this_frame           = imuFrame.imu_data_since_last_frame.back();
                    imuFrame.imu_directly_after_this_frame.timestamp = imuFrame.timestamp;
                }
            }

            imuFrame.computeInterpolatedImuValue();
            auto imu = getIMU();
            if (imu)
            {
                imuFrame.sanityCheck(imu.value());
            }
        }
    }

    Imu::Frame ImuDataForFrame(int frame) override
    {
        if (frame < imuDataForFrame.size())
        {
            return imuDataForFrame[frame];
        }
        else
        {
            return {};
        }
    }

    virtual std::optional<Imu::Sensor> getIMU() override
    {
        return imuData.empty() ? std::optional<Imu::Sensor>() : imu;
    }

   protected:
    AlignedVector<FrameType> frames;
    DatasetParameters params;

    Imu::Sensor imu;
    std::vector<Imu::Data> imuData;
    std::vector<Imu::Frame> imuDataForFrame;

   private:
    Timer timer;
    tick_t timeStep;
    tick_t lastFrameTime;
    tick_t nextFrameTime;
};



}  // namespace Saiga
