/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/icp/ICPAlign.h"
#include "saiga/vision/slam/Trajectory.h"
#include "saiga/vision/util/Random.h"

using namespace Saiga;

class RegistrationTest
{
   public:
    RegistrationTest(int N)
    {
        // Get random ground truth
        groundTruthTransformation      = Random::randomSE3();
        groundTruthTransformationScale = DSim3(groundTruthTransformation, gtscale);


        std::cout << groundTruthTransformation << std::endl;
        std::cout << groundTruthTransformationScale << std::endl;
        std::cout << groundTruthTransformationScale << std::endl;
        std::cout << std::endl;

        std::cout << groundTruthTransformationScale.inverse() << std::endl;


        std::cout << std::endl;
        Quat eigR = groundTruthTransformationScale.se3().unit_quaternion();
        Vec3 eigt = groundTruthTransformationScale.se3().translation();
        double s  = groundTruthTransformationScale.scale();
        eigt *= (1. / s);  //[R t/s;0 1]

        SE3 correctedTiw = SE3(eigR, eigt);

        std::cout << correctedTiw << std::endl;
        std::cout << correctedTiw.inverse() << std::endl;
        std::cout << groundTruthTransformationScale << std::endl;


        std::terminate();


        // create random point cloud
        // and transform by inverse ground truth transformation
        AlignedVector<Vec3> points, pointsTransformed, pointsTransformedScaled;
        auto invT  = groundTruthTransformation.inverse();
        auto invTs = groundTruthTransformationScale.inverse();
        for (int i = 0; i < N; ++i)
        {
            Vec3 p = Vec3::Random();
            points.push_back(p);
            pointsTransformed.push_back(invT * p);
            pointsTransformedScaled.push_back(invTs * p);
        }


        // create saiga icp corrs
        for (int i = 0; i < N; ++i)
        {
            ICP::Correspondence c;
            c.refPoint = points[i];

            c.srcPoint = pointsTransformed[i];
            pointCloud.push_back(c);

            c.srcPoint = pointsTransformedScaled[i];
            pointCloudScaled.push_back(c);
        }

        // create saiga trajectory
        for (int i = 0; i < N; ++i)
        {
            SE3 r           = Random::randomSE3();
            r.translation() = points[i];
            DSim3 rs        = DSim3(r, 1.0);

            gt.emplace_back(i, r);
            tracking.emplace_back(i, invT * r);

            DSim3 transformedSe3 = invTs * rs;
            SE3 backToSe3        = transformedSe3.se3();

            trackingScaled.emplace_back(i, backToSe3);
        }
    }

    void pointCloudSE3()
    {
        std::cout << "Testing SE3 registration..." << std::endl;
        auto result = ICP::pointToPointDirect(pointCloud);

        auto et = translationalError(groundTruthTransformation, result);
        auto rt = rotationalError(groundTruthTransformation, result);

        std::cout << "Error T/R: " << et << " " << rt << std::endl;
        std::cout << std::endl;
    }

    void pointCloudSim3()
    {
        std::cout << "Testing Sim3 registration..." << std::endl;
        double scale;
        auto result = ICP::pointToPointDirect(pointCloudScaled, &scale);


        auto et = translationalError(groundTruthTransformationScale.se3(), result);
        auto rt = rotationalError(groundTruthTransformationScale.se3(), result);
        auto es = std::abs(groundTruthTransformationScale.scale() - scale);

        std::cout << "Error T/R/S: " << et << " " << rt << " " << es << std::endl;
        std::cout << std::endl;
    }

    void trajectorySE3()
    {
        std::cout << "Testing SE3 trajectory alignment..." << std::endl;
        Trajectory::align(gt, tracking, false);

        auto ate = Statistics(Trajectory::ate(gt, tracking)).max;
        auto rpe = Statistics(Trajectory::rpe(gt, tracking, 1)).max;
        std::cout << "ATE/RPE " << ate << " " << rpe << std::endl;
        std::cout << std::endl;
    }

    void trajectorySim3()
    {
        std::cout << "Testing Sim3 trajectory alignment..." << std::endl;
        Trajectory::align(gt, trackingScaled, true);
        auto ate = Statistics(Trajectory::ate(gt, trackingScaled)).max;
        auto rpe = Statistics(Trajectory::rpe(gt, trackingScaled, 1)).max;
        std::cout << "ATE/RPE " << ate << " " << rpe << std::endl;
        std::cout << std::endl;
    }


    double gtscale = 5;
    SE3 groundTruthTransformation;
    AlignedVector<ICP::Correspondence> pointCloud;

    DSim3 groundTruthTransformationScale;
    AlignedVector<ICP::Correspondence> pointCloudScaled;


    Trajectory::TrajectoryType gt, tracking, trackingScaled;
};

int main(int, char**)
{
    RegistrationTest test(500);
    test.pointCloudSE3();
    test.pointCloudSim3();
    test.trajectorySE3();
    test.trajectorySim3();
    std::cout << "Done." << std::endl;
    return 0;
}
