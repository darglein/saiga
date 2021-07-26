/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



/**
 * The OpenCV distortion model applied to a point in normalized image coordinates.
 * Based on the distortion model given in vision/cameraModel/Distortion.h
 * Note: the 8-parameter model is subdivided into 2 vec4 coefficients
 */
vec2 distortNormalizedPoint(vec2 point, vec4 dis_1, vec4 dis_2)
{
    float x = point[0];
    float y = point[1];

    float k1 = dis_1[0];
    float k2 = dis_1[1];
    float k3 = dis_1[2];
    float k4 = dis_1[3];
    float k5 = dis_2[0];
    float k6 = dis_2[1];
    float p1 = dis_2[2];
    float p2 = dis_2[3];

    float x2 = x * x;
    float y2 = y * y;
    float r2 = x2 + y2;
    float _2xy = float(2) * x * y;

    if (r2 > 2){
        // The forward distortion fails if the points are too far away on the image plain
        return vec2(948435, 616);
    }

    float radial_u = float(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    float radial_v = float(1) + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
    float radial   = (radial_u / radial_v);

    float tangentialX = p1 * _2xy + p2 * (r2 + float(2) * x2);
    float tangentialY = p1 * (r2 + float(2) * y2) + p2 * _2xy;

    float xd = x * radial + tangentialX;
    float yd = y * radial + tangentialY;
    return vec2(xd, yd);
}