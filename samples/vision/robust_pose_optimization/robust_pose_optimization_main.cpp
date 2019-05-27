/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/RobustPoseOptimization.h"

#include <fstream>

using namespace Saiga;

class RPOTest
{
   public:
    RPOTest()
    {
        std::ifstream strm(SearchPathes::data("vision/poseRefinement.dat"), std::ios::binary);
        SAIGA_ASSERT(strm.is_open());

        int wpc;
        int obsc;
        strm.read((char*)&wpc, sizeof(int));
        strm.read((char*)&obsc, sizeof(int));
        strm.read((char*)&K, sizeof(StereoCamera4));
        strm.read((char*)pose.data(), sizeof(SE3));
        wps.resize(wpc);
        obs.resize(obsc);
        outlier.resize(wpc);
        for (auto& wp : wps) strm.read((char*)wp.data(), sizeof(Vec3));
        for (auto& ob : obs) strm.read((char*)&ob, sizeof(Obs));

        cout << wpc << " " << obsc << endl;
        optimize();
    }

    void optimize()
    {
        int inliers;
        {
            SAIGA_BLOCK_TIMER();
            inliers = rpo.optimizePoseRobust(wps, obs, outlier, pose, K);
        }

        cout << "[PoseRefinement] Initial/Final/Wps "
             << "/" << inliers << endl;
    }

    using T       = double;
    using SE3Type = Sophus::SE3<T>;
    using Vec3    = Eigen::Matrix<T, 3, 1>;
    using Vec2    = Eigen::Matrix<T, 2, 1>;
    using Obs     = ObsBase<T>;
    StereoCamera4 K;
    SE3 pose;
    AlignedVector<Vec3> wps;
    AlignedVector<Obs> obs;
    AlignedVector<bool> outlier;
    RobustPoseOptimization<T> rpo;
};


int main(int, char**)
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    Saiga::Random::setSeed(93865023985);

    RPOTest test;
    return 0;
}
