/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"


namespace Saiga
{
template <typename T>
HD Vector<T, 2> ProjectOCam(Vector<T, 3> p, Vector<T, 5> coeff_affine, ArrayView<T> coeff_poly)
{
    using Vec2 = Vector<T, 2>;

    T c  = coeff_affine(0);
    T d  = coeff_affine(1);
    T e  = coeff_affine(2);
    T cx = coeff_affine(3);
    T cy = coeff_affine(4);

    T norm  = sqrt(p[0] * p[0] + p[1] * p[1]);
    T theta = atan(p[2] / norm);

    if (norm < 1e-6)
    {
        return Vec2(cx, cy);
    }


    T invnorm = 1 / norm;
    T t       = theta;
    T rho     = coeff_poly[0];
    int t_i   = 1;

    for (int i = 1; i < coeff_poly.size(); i++)
    {
        t_i *= t;
        rho += t_i * coeff_poly[i];
    }

    T x = p[0] * invnorm * rho;
    T y = p[1] * invnorm * rho;

    Vec2 res;
    res[0] = x * c + y * d + cx;
    res[1] = x * e + y + cy;
    return res;
}

// Based on: Omnidirectional Camera Calibration Toolbox
// https://sites.google.com/site/scarabotix/ocamcalib-toolbox
//
// This is a fisheye camera model, for wide-angle cameras.
//
template <typename T>
struct OCam
{
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec2 = Eigen::Matrix<T, 2, 1>;

    int w = 0, h = 0;
    T c  = 1;
    T d  = 0;
    T e  = 0;
    T cx = 0;
    T cy = 0;
    Eigen::Matrix<T, -1, 1> poly_cam2world;
    Eigen::Matrix<T, -1, 1> poly_world2cam;

    OCam() {}

    OCam(int w, int h, Eigen::Matrix<T, 5, 1> ap, Eigen::Matrix<T, -1, 1> poly_cam2world,
         Eigen::Matrix<T, -1, 1> poly_world2cam)
        : w(w), h(h), poly_cam2world(poly_cam2world), poly_world2cam(poly_world2cam)
    {
        SetAffineParams(ap);
    }

    Eigen::Matrix<T, -1, 1> ProjectParams()
    {
        Eigen::Matrix<T, -1, 1> res;
        res.resize(NumProjectParams());
        res(0) = c;
        res(1) = d;
        res(2) = e;
        res(3) = cx;
        res(4) = cy;
        for (int i = 0; i < poly_world2cam.size(); ++i)
        {
            res(5 + i) = poly_world2cam[i];
        }
        return res;
    }

    int NumProjectParams() const { return 5 + poly_world2cam.size(); }
    int NumUnProjectParams() const { return 5 + poly_cam2world.size(); }

    Eigen::Matrix<T, 5, 1> AffineParams() const
    {
        Eigen::Matrix<T, 5, 1> p;
        p << c, d, e, cx, cy;
        return p;
    }
    void SetAffineParams(Eigen::Matrix<T, 5, 1> par)
    {
        c  = par(0);
        d  = par(1);
        e  = par(2);
        cx = par(3);
        cy = par(4);
    }

    void SetCam2World(ArrayView<T> params)
    {
        poly_cam2world.resize(params.size());
        for (int i = 0; i < params.size(); ++i)
        {
            poly_cam2world(i) = params[i];
        }
    }

    void SetWorld2Cam(ArrayView<T> params)
    {
        poly_world2cam.resize(params.size());
        for (int i = 0; i < params.size(); ++i)
        {
            poly_world2cam(i) = params[i];
        }
    }

    template <typename G>
    OCam<G> cast()
    {
        return OCam<G>(w, h, AffineParams().template cast<G>(), poly_cam2world.template cast<G>(),
                       poly_world2cam.template cast<G>());
    }


    // The project/unproject functions are inspired by
    // https://github.com/stonear/OCamCalib/blob/master/ocam_functions.h
    Vec2 Project(Vec3 p) const { return ProjectOCam(p, AffineParams(), poly_world2cam); }

    Vec3 InverseProject(Vec2 p, T depth = 1) const
    {
        T invdet = 1 / (c - d * e);
        T xp     = invdet * ((p[0] - cx) - d * (p[1] - cy));
        T yp     = invdet * (-e * (p[0] - cx) + c * (p[1] - cy));

        T r   = sqrt(xp * xp + yp * yp);
        T zp  = poly_cam2world[0];
        T r_i = 1;

        for (int i = 1; i < poly_cam2world.size(); i++)
        {
            r_i *= r;
            zp += r_i * poly_cam2world[i];
        }

        Vec3 result(xp, yp, 1);
        return result * depth;
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& strm, const OCam<T> intr)
{
    strm << intr.w << "x" << intr.h << "(" << intr.c << ", " << intr.d << ", " << intr.e << ", " << intr.cx << ", "
         << intr.cy << ") (";
    for (auto d : intr.poly_cam2world)
    {
        strm << d << ", ";
    }
    strm << ") (";
    for (auto d : intr.poly_world2cam)
    {
        strm << d << ", ";
    }
    strm << ")";
    return strm;
}


}  // namespace Saiga
