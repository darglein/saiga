/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "CameraBase.h"

#include "saiga/core/Core.h"
namespace Saiga
{
void DatasetParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    auto group = "Dataset";
    INI_GETADD_DOUBLE(ini, group, playback_fps);
    INI_GETADD_STRING(ini, group, dir);
    INI_GETADD_LONG(ini, group, startFrame);
    INI_GETADD_LONG(ini, group, maxFrames);
    INI_GETADD_BOOL(ini, group, multiThreadedLoad);
    INI_GETADD_BOOL(ini, group, preload);
    INI_GETADD_BOOL(ini, group, normalize_timestamps);
    INI_GETADD_DOUBLE(ini, group, ground_truth_time_offset);
    if (ini.changed()) ini.SaveFile(file.c_str());
}

std::ostream& operator<<(std::ostream& strm, const DatasetParameters& params)
{
    strm << params.dir;
    return strm;
}

DatasetCameraBase::DatasetCameraBase(const DatasetParameters& params) : params(params)
{
    timeStep =
        std::chrono::duration_cast<tick_t>(std::chrono::duration<double, std::micro>(1000000.0 / params.playback_fps));
    ResetTime();
}

void DatasetCameraBase::ResetTime()
{
    timer.start();
    lastFrameTime = timer.stop();
    nextFrameTime = lastFrameTime + timeStep;
}

void DatasetCameraBase::Load()
{
    SAIGA_ASSERT(this->camera_type != CameraInputType::Unknown);

    int num_images = LoadMetaData();
    SAIGA_ASSERT((int)frames.size() == num_images);
    //        frames.resize(num_images);
    computeImuDataPerFrame();

    if (params.preload)
    {
        ProgressBar loadingBar(std::cout, "Loading " + std::to_string(num_images) + " images ",
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

bool DatasetCameraBase::getImageSync(FrameData& data)
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

void DatasetCameraBase::saveGroundTruthTrajectory(const std::string& file)
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
            strm << time << " " << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z()
                 << " " << q.w() << std::endl;
        }
    }
}

void DatasetCameraBase::eraseFrames(int from, int to)
{
    frames.erase(frames.begin() + from, frames.begin() + to);
    imuDataForFrame.erase(imuDataForFrame.begin() + from, imuDataForFrame.begin() + to);
}

void DatasetCameraBase::computeImuDataPerFrame()
{
    // Create IMU per frame vector by adding all imu datas from frame_i to frame_i+1 to frame_i+1.
    imuDataForFrame.resize(frames.size());
    size_t currentImuid = 0;

    // Initialize imu sequences
    for (size_t i = 0; i < frames.size(); ++i)
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


    Imu::InterpolateMissingValues(imuDataForFrame);


    for (size_t i = 0; i < frames.size(); ++i)
    {
        frames[i].imu_data = imuDataForFrame[i];
    }
}

}  // namespace Saiga
