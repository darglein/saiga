/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "BALDataset.h"

#include "saiga/util/assert.h"
#include "saiga/util/file.h"
#include "saiga/util/tostring.h"

#include <fstream>

#ifdef SAIGA_USE_CERES
#    include "saiga/vision/ceres/CeresBAL.h"

#    include "ceres/ceres.h"
#endif

namespace Saiga
{
BALDataset::BALDataset(const std::string& file)
{
    cout << "> Loading BALDataset " << file << endl;

    auto data = File::loadFileStringArray(file);

    cout << "lines " << data.size() << endl;
    //    cout << "File loaded from memory: " << str.size() / (1000.0 * 1000.0) << "mb" << endl;

    int num_cameras, num_points, num_observations;

    std::stringstream in(data.front());
    in >> num_cameras >> num_points >> num_observations;

    cameras.resize(num_cameras);
    observations.resize(num_observations);
    points.resize(num_points);



    int start = 1;
#pragma omp parallel for
    for (int i = 0; i < num_observations; ++i)
    {
        std::stringstream in(data[start + i]);
        //        cout << data[start + i] << endl;
        BALObservation o;
        in >> o.camera_index >> o.point_index >> o.point[0] >> o.point[1];
        observations[i] = (o);

		if (i == 0)
        {
           //         cout << o.camera_index << " " << o.point_index << " " << o.point.transpose() << endl;
            }
    }

    start += num_observations;
#pragma omp parallel for
    for (int i = 0; i < num_cameras; ++i)
    {
        //        std::stringstream in(data[start + i]);
        //        cout << data[start + i] << endl;
        BALCamera c;
        Vec3 r;
        Vec3 t;


        r(0) = Saiga::to_double(data[start + i * 9 + 0]);
        r(1) = Saiga::to_double(data[start + i * 9 + 1]);
        r(2) = Saiga::to_double(data[start + i * 9 + 2]);

        t(0) = Saiga::to_double(data[start + i * 9 + 3]);
        t(1) = Saiga::to_double(data[start + i * 9 + 4]);
        t(2) = Saiga::to_double(data[start + i * 9 + 5]);

        c.f  = Saiga::to_double(data[start + i * 9 + 6]);
        c.k1 = Saiga::to_double(data[start + i * 9 + 7]);
        c.k2 = Saiga::to_double(data[start + i * 9 + 8]);


        //        in >> r(0) >> r(1) >> r(2) >> t(0) >> t(1) >> t(2) >> c.f >> c.k1 >> c.k2;
        auto angle           = r.norm();
        Eigen::Vector3d axis = angle > 0.00001 ? r / angle : Eigen::Vector3d(0, 1, 0);
        Eigen::AngleAxis<double> a(angle, axis);
        c.se3      = SE3((Quat)a, t);
        cameras[i] = (c);

		if (i == 0)
        {
          //          cout << c.se3 << " " << c.f << " " << c.k1 << " " << c.k2 << endl;       
        }
    }
    start += num_cameras * 9;

#pragma omp parallel for
    for (int i = 0; i < num_points; ++i)
    {
        //        std::stringstream in(data[start + i]);
        //        cout << data[start + i] << endl;
        BALPoint p;
        //        in >> p.point(0) >> p.point(1) >> p.point(2);

        p.point(0) = Saiga::to_double(data[start + i * 3 + 0]);
        p.point(1) = Saiga::to_double(data[start + i * 3 + 1]);
        p.point(2) = Saiga::to_double(data[start + i * 3 + 2]);


        points[i] = (p);

			if (i == 0)
        {
          //  cout << p.point.transpose() << endl;
        }
    }

    cout << "> Done. num_cameras " << num_cameras << " num_points " << num_points << " num_observations "
         << num_observations << endl;
    cout << "> RMS: " << rms() << endl;
    undistortAll();
    cout << "> RMS: " << rms() << endl;
}

void BALDataset::undistortAll()
{
#pragma omp parallel for
    for (int i = 0; i < (int)observations.size(); ++i)
    {
        BALObservation& o = observations[i];
        BALCamera c       = cameras[o.camera_index];
        o.point           = c.undistort(o.point);
    }


#ifdef SAIGA_USE_CERES1
    {
        cout << "Creating the ceres undistortion problem..." << endl;
        ceres::Problem problem;
        std::vector<Vec2> undistortedPoints(observations.size());
        for (int i = 0; i < observations.size(); ++i)
        {
            BALObservation& o = observations[i];
            BALCamera& c      = cameras[o.camera_index];

            auto& up           = undistortedPoints[i];
            up                 = o.point;
            auto cost_function = CostBALDistortion::create(o.point, c.k1, c.k2);
            problem.AddResidualBlock(cost_function, nullptr, up.data());
        }
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations           = 15;
        options.num_threads                  = 8;
        ceres::Solver::Summary summaryTest;
        ceres::Solve(options, &problem, &summaryTest);
        for (int i = 0; i < observations.size(); ++i)
        {
            BALObservation& o = observations[i];
            auto& up          = undistortedPoints[i];
            o.point           = up;
        }
    }
#endif

    for (BALCamera& c : cameras)
    {
        c.k1 = 0;
        c.k2 = 0;
    }
    cout << "Observations undistorted." << endl;
}

double BALDataset::rms()
{
    double error = 0;
//    for (BALObservation& o : observations)
#pragma omp parallel for reduction(+:error)
    for (int i = 0; i < (int)observations.size(); ++i)
    {
        BALObservation& o = observations[i];
        BALCamera c       = cameras[o.camera_index];
        BALPoint p        = points[o.point_index];

        Eigen::Vector2d projPoint = c.projectPoint(p.point);

        auto sqerror = (projPoint - o.point).squaredNorm();

        error += sqerror;
    }
    error /= observations.size();
    error = sqrt(error);
    return error;
}

Scene BALDataset::makeScene()
{
    Scene scene;
    std::vector<double> fs;
    for (BALCamera& c : cameras)
    {
        int id = scene.images.size();

        SceneImage si;
        si.intr = id;
        si.extr = id;
        scene.images.push_back(si);
        scene.intrinsics.push_back(c.intr());
        scene.extrinsics.push_back(c.extr());
        fs.push_back(c.f);
    }

    for (BALObservation& o : observations)
    {
        SAIGA_ASSERT(o.camera_index >= 0 && o.camera_index < (int)scene.images.size());
        SceneImage& i = scene.images[o.camera_index];
        i.stereoPoints.push_back(o.ip());
    }

    for (BALPoint& p : points)
    {
        scene.worldPoints.push_back(p.wp());
    }


    // the datasets already have an reasonable scale
    scene.globalScale = 1;


    scene.fixWorldPointReferences();
    SAIGA_ASSERT(scene.valid());

    cout << "Created a Saiga::Scene from BAL dataset." << endl;

    return scene;
}

}  // namespace Saiga
