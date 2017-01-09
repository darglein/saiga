#pragma once

#include <Eigen/Core>


//Note: The DontAlign option disables all vectorisation.
typedef Eigen::Matrix<float,2,1,Eigen::DontAlign> vec2_t;
typedef Eigen::Matrix<float,3,1,Eigen::DontAlign> vec3_t;
typedef Eigen::Matrix<float,4,1,Eigen::DontAlign> vec4_t;
typedef Eigen::Matrix<float,3,3,Eigen::DontAlign> mat3_t;
typedef Eigen::Matrix<float,4,4,Eigen::DontAlign> mat4_t;
typedef Eigen::Quaternion<float,Eigen::DontAlign> quat_t;


typedef Eigen::Matrix<double,2,1,Eigen::DontAlign> vec2d_t;
typedef Eigen::Matrix<double,3,1,Eigen::DontAlign> vec3d_t;
typedef Eigen::Matrix<double,4,1,Eigen::DontAlign> vec4d_t;
typedef Eigen::Matrix<double,3,3,Eigen::DontAlign> mat3d_t;
typedef Eigen::Matrix<double,4,4,Eigen::DontAlign> mat4d_t;
typedef Eigen::Quaternion<double,Eigen::DontAlign> quatd_t;
