#pragma once
#include "saiga/core/math/math.h"

#include "torch/torch.h"


#if TORCH_VERSION_MAJOR > 1 || TORCH_VERSION_MINOR >= 11
// This is a helper function so that we can use
// tensor.data_ptr<vec3>() on tensors with the shape
//   [x,x,...,x,3]
template <>
inline Saiga::vec3* at::TensorBase::data_ptr<Saiga::vec3>() const
{
    CHECK_EQ(size(dim() - 1), 3);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::vec3*)data_ptr<float>();
}
template <>
inline Saiga::vec2* at::TensorBase::data_ptr<Saiga::vec2>() const
{
    CHECK_EQ(size(dim() - 1), 2);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::vec2*)data_ptr<float>();
}
template <>
inline Saiga::Quat* at::TensorBase::data_ptr<Saiga::Quat>() const
{
    CHECK_EQ(size(dim() - 1), 4);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::Quat*)data_ptr<double>();
}

template <>
inline Saiga::Vec3* at::TensorBase::data_ptr<Saiga::Vec3>() const
{
    CHECK_EQ(size(dim() - 1), 3);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::Vec3*)data_ptr<double>();
}
template <>
inline Saiga::Vec2* at::TensorBase::data_ptr<Saiga::Vec2>() const
{
    CHECK_EQ(size(dim() - 1), 2);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::Vec2*)data_ptr<double>();
}
#else
// This is a helper function so that we can use
// tensor.data_ptr<vec3>() on tensors with the shape
//   [x,x,...,x,3]
template <>
inline Saiga::vec3* at::Tensor::data_ptr<Saiga::vec3>() const
{
    CHECK_EQ(size(dim() - 1), 3);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::vec3*)data_ptr<float>();
}
template <>
inline Saiga::vec2* at::Tensor::data_ptr<Saiga::vec2>() const
{
    CHECK_EQ(size(dim() - 1), 2);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::vec2*)data_ptr<float>();
}
template <>
inline Saiga::Quat* at::Tensor::data_ptr<Saiga::Quat>() const
{
    CHECK_EQ(size(dim() - 1), 4);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::Quat*)data_ptr<double>();
}

template <>
inline Saiga::Vec3* at::Tensor::data_ptr<Saiga::Vec3>() const
{
    CHECK_EQ(size(dim() - 1), 3);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::Vec3*)data_ptr<double>();
}
template <>
inline Saiga::Vec2* at::Tensor::data_ptr<Saiga::Vec2>() const
{
    CHECK_EQ(size(dim() - 1), 2);
    CHECK_EQ(stride(dim() - 1), 1);
    return (Saiga::Vec2*)data_ptr<double>();
}
#endif