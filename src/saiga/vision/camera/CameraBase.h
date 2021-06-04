/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/time/timer.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/Thread/omp.h"

#include "CameraData.h"

#include <fstream>
#include <iomanip>
#include <thread>

namespace Saiga
{
/**
 * Interface class for different datset inputs.
 */
class SAIGA_TEMPLATE CameraBase
{
   public:
    virtual ~CameraBase() {}

    // Blocks until the next image is available
    // Returns true if success.
    virtual bool getImageSync(FrameData& data) = 0;

    // Returns false if no image is currently available
    virtual bool getImage(FrameData& data) { return getImageSync(data); }


    virtual void close() {}
    virtual bool isOpened() { return true; }


    virtual std::vector<std::pair<double, SE3>> GetGroundTruth() const { return {}; }

    std::optional<Imu::Sensor> getIMU() { return imu; }
    std::optional<Imu::Sensor> imu;

    CameraInputType CameraType() const { return camera_type; }

   protected:
    CameraInputType camera_type = CameraInputType::Unknown;
    int currentId               = 0;
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
class SAIGA_VISION_API DatasetCameraBase : public CameraBase
{
   public:
    DatasetCameraBase(const DatasetParameters& params);

    void ResetTime();

    void Load();

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

    virtual void LoadImageData(FrameData& data) {}



    bool getImageSync(FrameData& data) override;

    virtual bool isOpened() override { return this->currentId < (int)frames.size(); }
    size_t getFrameCount() { return frames.size(); }

    std::vector<std::pair<double, SE3>> GetGroundTruth() const override { return ground_truth; }


    // Saves the groundtruth in TUM-Trajectory format:
    // <timestamp> <translation x y z> <rotation x y z w>
    void saveGroundTruthTrajectory(const std::string& file);

    // Completely removes the frames between from and to
    void eraseFrames(int from, int to);

    void computeImuDataPerFrame();



   protected:
    AlignedVector<FrameData> frames;
    DatasetParameters params;
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
