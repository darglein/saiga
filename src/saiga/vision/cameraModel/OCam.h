/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"


namespace Saiga
{
// Important!!!
// This is the right-handed model, which is not identical to the original matlab implementation.
// In particular the input point is swapped in x-y direction and z is multiplied by -1.
// The output point is then swapped back.
template <typename T>
HD Vector<T, 3> ProjectOCam(Vector<T, 3> p, Vector<T, 5> coeff_affine, ArrayView<const T> coeff_poly,
                            float cutoff = 10000)
{
    using Vec3 = Vector<T, 3>;

    T c  = coeff_affine(0);
    T d  = coeff_affine(1);
    T e  = coeff_affine(2);
    T cx = coeff_affine(3);
    T cy = coeff_affine(4);

    // Coordinate system switch!!
    T x = p(1);
    T y = p(0);
    T z = -p(2);

    T norm  = sqrt(x * x + y * y);
    T theta = atan(z / norm);
    T dist  = p.norm();

    if (theta > cutoff)
    {
        return Vec3(0.f, 0.f, 0.f);
    }

    if (norm < 1e-6)
    {
        return Vec3(cx, cy, dist);
    }


    T invnorm = 1 / norm;
    T t       = theta;
    T rho     = coeff_poly[0];
    T t_i     = 1;

    for (int i = 1; i < coeff_poly.size(); i++)
    {
        t_i *= t;
        rho += t_i * coeff_poly[i];
    }

    T np_x = x * invnorm * rho;
    T np_y = y * invnorm * rho;


    T image_x = np_x * c + np_y * d + cx;
    T image_y = np_x * e + np_y + cy;

    // Again coordinate system switch!!
    return Vec3(image_y, image_x, dist);
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

    // Only the affine parameters are scaled!
    OCam<T> scale(T s) const
    {
        OCam<T> res = *this;
        res.SetAffineParams(AffineParams() * s);
        return res;
    }

    Eigen::Matrix<T, 5, 1> AffineParams() const
    {
        Eigen::Matrix<T, 5, 1> par;
        par(0) = c;
        par(1) = d;
        par(2) = e;
        par(3) = cx;
        par(4) = cy;
        return par;
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
    Vec2 Project(Vec3 p) const
    {
        ArrayView<const T> pv(poly_world2cam.data(), poly_world2cam.size());
        return ProjectOCam<T>(p, AffineParams(), pv).template head<2>();
    }

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
    strm << intr.w << "x" << intr.h << " affine(" << intr.c << ", " << intr.d << ", " << intr.e << ", " << intr.cx
         << ", " << intr.cy << ") cam2world(";
    for (auto d : intr.poly_cam2world)
    {
        strm << d << ", ";
    }
    strm << ") world2cam(";
    for (auto d : intr.poly_world2cam)
    {
        strm << d << ", ";
    }
    strm << ")";
    return strm;
}


}  // namespace Saiga
