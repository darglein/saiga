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
//
// The derivate is currently only available in respect to p.
//
// Return: [ Vec2(image_point), distance_to_camera ]
//
template <typename T>
HD Vector<T, 3> ProjectOCam(Vector<T, 3> p, Vector<T, 5> coeff_affine, ArrayView<const T> coeff_poly,
                            float cutoff = 10000, Matrix<T, 2, 3>* jacobian_point = nullptr,
                            Matrix<T, 2, 5>* jacobian_affine = nullptr)
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

    T norm2     = x * x + y * y;
    T norm      = sqrt(norm2);
    T invnorm   = 1 / norm;
    T z_by_norm = z * invnorm;
    T theta     = atan(z_by_norm);
    T dist      = p.norm();

    if (theta > cutoff)
    {
        return Vec3(0.f, 0.f, 0.f);
    }

    if (norm < 1e-6)
    {
        if (jacobian_point) jacobian_point->setZero();
        if (jacobian_affine) jacobian_affine->setZero();
        return Vec3(cx, cy, dist);
    }

    T rho = coeff_poly[0];
    T t_i = 1;

    for (int i = 1; i < coeff_poly.size(); i++)
    {
        t_i *= theta;
        rho += t_i * coeff_poly[i];
    }

    T np_x = x * invnorm * rho;
    T np_y = y * invnorm * rho;


    T image_x = np_x * c + np_y * d + cx;
    T image_y = np_x * e + np_y + cy;

    if (jacobian_point)
    {
        auto& J = *jacobian_point;

        // rho w.r.t theta
        T drho_dtheta = 0;
        T t_i         = 1;
        for (int i = 1; i < coeff_poly.size(); i++)
        {
            drho_dtheta += i * coeff_poly[i] * t_i;
            t_i *= theta;
        }

        // theta w.r.t x y z
        T xyz_norm_sqr = norm2 + z * z;
        T dtheta_dx    = (-1.0 * x * z_by_norm) / (xyz_norm_sqr);
        T dtheta_dy    = (-1.0 * y * z_by_norm) / (xyz_norm_sqr);
        T dtheta_dz    = norm / (xyz_norm_sqr);

        // rho w.r.t x y z
        T drho_dx = drho_dtheta * dtheta_dx;
        T drho_dy = drho_dtheta * dtheta_dy;
        T drho_dz = drho_dtheta * dtheta_dz;

        // uv_raw w.r.t x y z
        // J(0, 0)  = (norm - x * x / norm) / norm2 * rho + drho_dx * x / norm;
        J(0, 0) = (invnorm - x * x / (norm * norm2)) * rho + drho_dx * x / norm;
        J(0, 1) = (-1.0 * x * y / norm) / norm2 * rho + drho_dy * x / norm;
        J(0, 2) = drho_dz * x / norm;
        J(1, 0) = (-1.0 * x * y / norm) / norm2 * rho + drho_dx * y / norm;
        J(1, 1) = (norm - y * y / norm) / norm2 * rho + drho_dy * y / norm;
        J(1, 2) = drho_dz * y / norm;

        // Affine transformation
        Matrix<T, 2, 2> affine;
        affine(0, 0) = c;
        affine(0, 1) = d;
        affine(1, 0) = e;
        affine(1, 1) = 1;
        J            = affine * J;

        if (true)
        {
            // swap coordinates and negate z
            auto a = J(0, 0);
            auto b = J(0, 1);
            auto c = J(0, 2);

            J(0, 0) = J(1, 1);
            J(0, 1) = J(1, 0);
            J(0, 2) = J(1, 2);

            J(1, 0) = b;
            J(1, 1) = a;
            J(1, 2) = c;

            J(0, 2) = -J(0, 2);
            J(1, 2) = -J(1, 2);
        }
    }

    if (jacobian_affine)
    {
        auto& J = *jacobian_affine;
        J.setZero();

        // Warning this already contains the coordinate transform!!!
        J(1, 0) = np_x;
        J(1, 1) = np_y;
        J(0, 2) = np_x;
        J(0, 4) = 1;
        J(1, 3) = 1;
    }

    // Again coordinate system switch!!
    return Vec3(image_y, image_x, dist);
}

template <typename T>
HD Vector<T, 3> UnprojectOCam(Vector<T, 2> p, T distance_to_cam, Vector<T, 5> coeff_affine,
                              ArrayView<const T> coeff_poly)
{
    T c  = coeff_affine(0);
    T d  = coeff_affine(1);
    T e  = coeff_affine(2);
    T cx = coeff_affine(3);
    T cy = coeff_affine(4);

    // Coordinate system switch!!
    T image_x = p(1);
    T image_y = p(0);


    T invdet = 1 / (c - d * e);
    T xp     = invdet * ((image_x - cx) - d * (image_y - cy));
    T yp     = invdet * (-e * (image_x - cx) + c * (image_y - cy));

    T r   = sqrt(xp * xp + yp * yp);
    T zp  = coeff_poly[0];
    T r_i = 1;

    for (int i = 1; i < coeff_poly.size(); i++)
    {
        r_i *= r;
        zp += r_i * coeff_poly[i];
    }

    T inv_norm = 1 / sqrt(xp * xp + yp * yp + zp * zp);
    inv_norm *= distance_to_cam;

    T px = inv_norm * xp;
    T py = inv_norm * yp;
    T pz = inv_norm * zp;

    // Again coordinate system switch!!
    return {py, px, -pz};
}

// Based on: Omnidirectional Camera Calibration Toolbox
// https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab?authuser=0
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
    std::vector<T> poly_cam2world;
    std::vector<T> poly_world2cam;

    OCam() {}

    OCam(int w, int h, Eigen::Matrix<T, 5, 1> ap, ArrayView<T> _poly_cam2world, ArrayView<T> _poly_world2cam)
        : w(w), h(h)
    {
        SetAffineParams(ap);
        SetCam2World(_poly_cam2world);
        SetWorld2Cam(_poly_world2cam);
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

    Eigen::Matrix<T, 3, 3> AffineMatrix() const
    {
        Eigen::Matrix<T, 3, 3> res;
        res.setZero();

        res(0, 0) = c;
        res(0, 1) = d;
        res(0, 2) = cx;

        res(1, 1) = c;
        res(0, 1) = d;


        return res;
    }

    void SetCam2World(ArrayView<T> params)
    {
        poly_cam2world.resize(params.size());
        for (int i = 0; i < params.size(); ++i)
        {
            poly_cam2world[i] = params[i];
        }
    }

    void SetWorld2Cam(ArrayView<T> params)
    {
        poly_world2cam.resize(params.size());
        for (int i = 0; i < params.size(); ++i)
        {
            poly_world2cam[i] = params[i];
        }
    }

    template <typename G>
    OCam<G> cast()
    {
        std::vector<G> poly_cam2world_new;
        std::vector<G> poly_world2cam_new;

        for (auto v : poly_cam2world)
        {
            poly_cam2world_new.push_back(v);
        }

        for (auto v : poly_world2cam)
        {
            poly_world2cam_new.push_back(v);
        }

        return OCam<G>(w, h, AffineParams().template cast<G>(), poly_cam2world_new, poly_world2cam_new);
    }


    // The project/unproject functions are inspired by
    // https://github.com/stonear/OCamCalib/blob/master/ocam_functions.h
    Vec2 Project(Vec3 p) const
    {
        return ProjectOCam<T>(p, AffineParams(), poly_world2cam).template head<2>();
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
