/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "EigenGLMInterface.h"

#include "math.h"


namespace Saiga
{
mat4 createTRSmatrix(const vec3& t, const quat& r, const vec3& s)
{
    mat4 T = translate(t);
    mat4 R = make_mat4(r);
    mat4 S = scale(s);
    return T * R * S;
}

mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s)
{
    mat4 T = translate(make_vec3(t));
    mat4 R = make_mat4(r);
    mat4 S = scale(make_vec3(s));
    return T * R * S;
}

mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar)
{
    mat4 Result  = mat4::Identity();
    Result(0, 0) = static_cast<float>(2) / (right - left);
    Result(1, 1) = static_cast<float>(2) / (top - bottom);
    Result(2, 2) = -static_cast<float>(2) / (zFar - zNear);
    Result(0, 3) = -(right + left) / (right - left);
    Result(1, 3) = -(top + bottom) / (top - bottom);
    Result(2, 3) = -(zFar + zNear) / (zFar - zNear);
    return Result;
}

mat4 perspective(float fovy, float aspect, float zNear, float zFar)
{
    float const tanHalfFovy = tan(fovy / static_cast<float>(2));

    mat4 Result = mat4::Zero();

    // Make sure the FOV is defined for the longer side.
    // Otherwise we can get distorted images if the output is rectangular
    if (aspect > 1)
    {
        Result(0, 0) = static_cast<float>(1) / (tanHalfFovy);
        Result(1, 1) = static_cast<float>(1) / (tanHalfFovy / aspect);
    }
    else
    {
        Result(0, 0) = static_cast<float>(1) / (aspect * tanHalfFovy);
        Result(1, 1) = static_cast<float>(1) / (tanHalfFovy);
    }


    Result(2, 2) = -(zFar + zNear) / (zFar - zNear);
    Result(3, 2) = -static_cast<float>(1);
    Result(2, 3) = -(static_cast<float>(2) * zFar * zNear) / (zFar - zNear);
    return Result;
}

mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up)
{
    // right handed coordinate system

    vec3 const f(normalize((center - eye).eval()));
    vec3 const s(normalize(cross(f, up)));
    vec3 const u(cross(s, f));

    mat4 Result  = mat4::Identity();
    Result(0, 0) = s.x();
    Result(0, 1) = s.y();
    Result(0, 2) = s.z();
    Result(1, 0) = u.x();
    Result(1, 1) = u.y();
    Result(1, 2) = u.z();
    Result(2, 0) = -f.x();
    Result(2, 1) = -f.y();
    Result(2, 2) = -f.z();
    Result(0, 3) = -dot(s, eye);
    Result(1, 3) = -dot(u, eye);
    Result(2, 3) = dot(f, eye);
    return Result;
}

mat4 scale(const vec3& t)
{
    mat4 m2  = mat4::Identity();
    m2(0, 0) = t(0);
    m2(1, 1) = t(1);
    m2(2, 2) = t(2);
    return m2;
}

vec4 make_vec4(float x, float y, float z, float w)
{
    return vec4(x, y, z, w);
}

vec4 make_vec4(const vec3& v, float a)
{
    return vec4(v(0), v(1), v(2), a);
}

vec3 make_vec3(const vec2& v, float a)
{
    return vec3(v(0), v(1), a);
}

vec4 make_vec4(const vec2& v, const vec2& v2)
{
    return vec4(v(0), v(1), v2(0), v2(1));
}

vec4 make_vec4(float a)
{
    return vec4(a, a, a, a);
}

vec3 make_vec3(const vec4& a)
{
    return vec3(a(0), a(1), a(2));
}

vec3 make_vec3(float a)
{
    return vec3(a, a, a);
}

vec2 make_vec2(float a)
{
    return vec2(a, a);
}

vec2 make_vec2(const vec3& a)
{
    return vec2(a(0), a(1));
}

vec2 make_vec2(float a, float b)
{
    return vec2(a, b);
}

vec2 make_vec2(const ivec2& a)
{
    return a.cast<float>();
}

ivec2 make_ivec2(int a, int b)
{
    ivec2 v;
    v(0) = a;
    v(1) = b;
    return v;
}

ucvec4 make_ucvec4(const ucvec3& v, unsigned char a)
{
    return ucvec4(v(0), v(1), v(2), a);
}

mat4 make_mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13, float a20,
               float a21, float a22, float a23, float a30, float a31, float a32, float a33)
{
    mat4 m;
    m << a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33;
    return m.transpose();
}

mat4 make_mat4_row_major(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13,
                         float a20, float a21, float a22, float a23, float a30, float a31, float a32, float a33)
{
    mat4 m;
    m << a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33;
    return m;
}

mat4 make_mat4(const mat3& m)
{
    mat4 m4              = mat4::Identity();
    m4.block<3, 3>(0, 0) = m;
    return m4;
}

mat3 make_mat3(const mat4& m)
{
    return m.block<3, 3>(0, 0);
}

mat3 make_mat3(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
{
    mat3 m;
    m << a00, a01, a02, a10, a11, a12, a20, a21, a22;
    return m.transpose();
}

mat4 make_mat4(const quat& q)
{
    return make_mat4(q.matrix());
}

mat3 make_mat3(const quat& q)
{
    return q.matrix();
}

quat make_quat(float x, float y, float z, float w)
{
    // Eigen quats are stored as (x,y,z,w), but the constructor is (w,x,y,z)
    return quat(w, x, y, z);
}

vec4 quat_to_vec4(const quat& q)
{
    return vec4(q.x(), q.y(), q.z(), q.w());
}

quat make_quat(const mat3& m)
{
    return quat(m);
}

quat make_quat(const mat4& m)
{
    return make_quat(make_mat3(m));
}

quat angleAxis(float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return make_quat(aa.matrix());
}

quat mix(const quat& a, const quat& b, float alpha)
{
    return a.slerp(alpha, b);
}

Quat mix(const Quat& a, const Quat& b, double alpha)
{
    return a.slerp(alpha, b);
}

quat quat_cast(const mat3& m)
{
    return quat(m).normalized();
}

quat quat_cast(const mat4& m)
{
    return quat_cast(make_mat3(m));
}

quat inverse(const quat& q)
{
    return q.inverse();
}

mat4 rotate(float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return make_mat4(aa.matrix());
}

quat rotate(const quat& q, float angle, const vec3& axis)
{
    Eigen::AngleAxisf aa(angle, axis);
    return q * aa;
}



quat rotation(const vec3& a, const vec3& b)
{
    return quat::FromTwoVectors(a, b);
}

mat4 translate(const vec3& t)
{
    mat4 m              = mat4::Identity();
    m.block<3, 1>(0, 3) = t;
    return m;
}


}  // namespace Saiga
