#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigen>

//Note: The DontAlign option disables all vectorisation.

#define EIGEN_MATRIX_OPTIONS Eigen::DontAlign|Eigen::ColMajor
typedef Eigen::Matrix<float,2,1,EIGEN_MATRIX_OPTIONS> vec2_t;
typedef Eigen::Matrix<float,3,1,EIGEN_MATRIX_OPTIONS> vec3_t;
typedef Eigen::Matrix<float,4,1,EIGEN_MATRIX_OPTIONS> vec4_t;
typedef Eigen::Matrix<float,2,2,EIGEN_MATRIX_OPTIONS> mat2_t;
typedef Eigen::Matrix<float,3,3,EIGEN_MATRIX_OPTIONS> mat3_t;
typedef Eigen::Matrix<float,4,4,EIGEN_MATRIX_OPTIONS> mat4_t;
typedef Eigen::Quaternion<float,EIGEN_MATRIX_OPTIONS> quat_t;


typedef Eigen::Matrix<double,2,1,EIGEN_MATRIX_OPTIONS> vec2d_t;
typedef Eigen::Matrix<double,3,1,EIGEN_MATRIX_OPTIONS> vec3d_t;
typedef Eigen::Matrix<double,4,1,EIGEN_MATRIX_OPTIONS> vec4d_t;
typedef Eigen::Matrix<double,2,2,EIGEN_MATRIX_OPTIONS> mat2d_t;
typedef Eigen::Matrix<double,3,3,EIGEN_MATRIX_OPTIONS> mat3d_t;
typedef Eigen::Matrix<double,4,4,EIGEN_MATRIX_OPTIONS> mat4d_t;
typedef Eigen::Quaternion<double,EIGEN_MATRIX_OPTIONS> quatd_t;
