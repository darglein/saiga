/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "BALDataset.h"

#include "saiga/util/assert.h"

#include <fstream>
namespace Saiga
{
BALDataset::BALDataset(const std::string& file)
{
    cout << "> Loading BALDataset " << file << endl;
    std::ifstream in(file);
    SAIGA_ASSERT(in.is_open());

    int num_cameras, num_points, num_observations;

    in >> num_cameras >> num_points >> num_observations;



    for (int i = 0; i < num_observations; ++i)
    {
        BALObservation o;
        in >> o.camera_index >> o.point_index >> o.point.x() >> o.point.y();
        observations.push_back(o);
    }

    for (int i = 0; i < num_cameras; ++i)
    {
        BALCamera c;
        Vec3 r;
        Vec3 t;
        in >> r(0) >> r(1) >> r(2) >> t(0) >> t(1) >> t(2) >> c.f >> c.k1 >> c.k2;
        auto angle           = r.norm();
        Eigen::Vector3d axis = angle > 0.00001 ? r / angle : Eigen::Vector3d(0, 1, 0);
        Eigen::AngleAxis<double> a(angle, axis);
        c.se3 = SE3((Quat)a, t);
        cameras.push_back(c);
    }

    for (int i = 0; i < num_points; ++i)
    {
        BALPoint p;
        in >> p.point(0) >> p.point(1) >> p.point(2);
        points.push_back(p);
    }

    cout << "> Done. num_cameras " << num_cameras << " num_points " << num_points << " num_observations "
         << num_observations << endl;
    cout << "> RMS: " << rms() << endl;
    undistortAll();
    cout << "> RMS: " << rms() << endl;
}

void BALDataset::undistortAll()
{
    for (BALObservation& o : observations)
    {
        BALCamera c = cameras[o.camera_index];
        o.point     = c.undistort(o.point);
    }

    for (BALCamera& c : cameras)
    {
        c.k1 = 0;
        c.k2 = 0;
    }
}

double BALDataset::rms()
{
    double error = 0;
    for (BALObservation& o : observations)
    {
        BALCamera c = cameras[o.camera_index];
        BALPoint p  = points[o.point_index];

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
    for (BALCamera& c : cameras)
    {
        int id = scene.images.size();

        SceneImage si;
        si.intr = id;
        si.extr = id;
        scene.images.push_back(si);
        scene.intrinsics.push_back(c.intr());
        scene.extrinsics.push_back(c.extr());
    }

    for (BALObservation& o : observations)
    {
        SAIGA_ASSERT(o.camera_index >= 0 && o.camera_index < (int)scene.images.size());
        SceneImage& i = scene.images[o.camera_index];
        i.monoPoints.push_back(o.ip());
    }

    for (BALPoint& p : points)
    {
        scene.worldPoints.push_back(p.wp());
    }

    scene.fixWorldPointReferences();
    SAIGA_ASSERT(scene.valid());

    return scene;
}

}  // namespace Saiga
