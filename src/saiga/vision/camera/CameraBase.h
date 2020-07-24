/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/time/timer.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/Thread/omp.h"
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


    virtual std::vector<std::pair<double, SE3>> GetGroundTruth() const { return {}; }

    // Optional IMU data if the camera provides it.
    // The returned vector contains all data from frame-1 to frame.
    virtual Imu::ImuSequence ImuDataForFrame(int frame) { return {}; }
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

    // Throw away all frames before 'startFrame'
    int startFrame = 0;

    // Only load 'maxFrames' after the startFrame.
    int maxFrames = -1;

    // Load images in parallel with omp
    bool multiThreadedLoad = true;

    // Force monocular load. Don't load second image or depth map for stereo and rgbd sensors.
    bool force_monocular = false;

    // Load all images to ram at the beginning.
    bool preload = true;

    // Subtract the timestamp of the first image from everything.
    bool normalize_timestamps = false;

    // Time offset added to ground truth trajectory.
    // Can be used if the ground truth data was synchronized badly.
    double ground_truth_time_offset = 0;

    void fromConfigFile(const std::string& file);

    friend std::ostream& operator<<(std::ostream& strm, const DatasetParameters& params);
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
        ResetTime();
    }

    void ResetTime()
    {
        timer.start();
        lastFrameTime = timer.stop();
        nextFrameTime = lastFrameTime + timeStep;
    }

    void Load()
    {
        int num_images = LoadMetaData();
        SAIGA_ASSERT(frames.size() == num_images);
        //        frames.resize(num_images);
        computeImuDataPerFrame();

        if (params.preload)
        {
            SyncedConsoleProgressBar loadingBar(std::cout, "Loading " + std::to_string(num_images) + " images ",
                                                num_images);

            // More than 8 doesn't improve performance even on NVME SSDs.
            int threads = std::min<int>(OMP::getMaxThreads(), 8);

#pragma omp parallel for if (params.multiThreadedLoad) num_threads(threads)
            for (int i = 0; i < num_images; ++i)
            {
                LoadImageData(frames[i]);
                loadingBar.addProgress(1);
            }
        }
        ResetTime();
    }

    // Load the dataset meta data. This information must be stored in the derived loader class, because it can differ
    // between datasets.
    //  - Intrinsics
    //  - Ground Truth
    //  - Image Names - timestamps
    //  - IMU measurements
    //  -  ...
    //
    // Returns the number of images.
    virtual int LoadMetaData() { return 0; }

    virtual void LoadImageData(FrameType& data) {}



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
        if (!params.preload)
        {
            LoadImageData(img);
        }
        this->currentId++;
        data = std::move(img);
        return true;
    }

    virtual bool isOpened() override { return this->currentId < (int)frames.size(); }
    size_t getFrameCount() { return frames.size(); }

    std::vector<std::pair<double, SE3>> GetGroundTruth() const override { return ground_truth; }


    // Saves the groundtruth in TUM-Trajectory format:
    // <timestamp> <translation x y z> <rotation x y z w>
    void saveGroundTruthTrajectory(const std::string& file)
    {
        std::ofstream strm(file);
        strm << std::setprecision(20);
        for (auto& f : frames)
        {
            if (f.groundTruth)
            {
                double time = f.timeStamp;
                SE3 pose    = f.groundTruth.value();
                Vec3 t      = pose.translation();
                Quat q      = pose.unit_quaternion();
                strm << time << " " << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " "
                     << q.z() << " " << q.w() << std::endl;
            }
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

        // Initialize imu sequences
        for (int i = 0; i < frames.size(); ++i)
        {
            Imu::ImuSequence& imuFrame = imuDataForFrame[i];

            auto& frame       = frames[i];
            imuFrame.time_end = frame.timeStamp;
            for (; currentImuid < imuData.size(); ++currentImuid)
            {
                auto id = imuData[currentImuid];
                if (id.timestamp <= frame.timeStamp)
                {
                    imuFrame.data.push_back(id);
                }
                else
                {
                    break;
                }
            }


            if (i >= 1)
            {
                imuFrame.time_begin = imuDataForFrame[i - 1].time_end;
            }
        }



        if (!imuDataForFrame.empty())
        {
            // Clear first frame.
            // It doesn't really make much sense because it contains the data from the previous to this frame.
            //            imuDataForFrame.front() = Imu::ImuSequence();
        }

        Imu::InterpolateMissingValues(imuDataForFrame);
    }

    Imu::ImuSequence ImuDataForFrame(int frame) override
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
    std::vector<Imu::ImuSequence> imuDataForFrame;

    std::vector<std::pair<double, SE3>> ground_truth;

   private:
    Timer timer;
    tick_t timeStep;
    tick_t lastFrameTime;
    tick_t nextFrameTime;
};



}  // namespace Saiga
