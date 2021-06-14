/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/util/ini/ini.h"

#include "ImageTensor.h"
#include "PartialConv.h"
#include "TorchHelper.h"

namespace Saiga
{
class UnetBlockImpl : public torch::nn::Module
{
   public:
    // Takes an image + mask and returns and updated image and updated mask
    virtual std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) = 0;

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> x_mask)
    {
        return forward(x_mask.first, x_mask.second);
    }
};
TORCH_MODULE(UnetBlock);

class BasicBlockImpl : public UnetBlockImpl
{
   public:
    BasicBlockImpl(int in_channels, int out_channels, int kernel_size = 3, std::string norm_str = "bn")
    {
        block->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(1)));
        block->push_back(NormFromString(norm_str, out_channels));
        block->push_back(torch::nn::ReLU());

        block->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).padding(1)));
        block->push_back(NormFromString(norm_str, out_channels));
        block->push_back(torch::nn::ReLU());

        register_module("block", block);
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        auto res = block->forward(x);
        return {res, mask};
    }

    torch::nn::Sequential block;
};

TORCH_MODULE(BasicBlock);


class PartialBlockImpl : public UnetBlockImpl
{
   public:
    PartialBlockImpl(int in_channels, int out_channels, int kernel_size = 3, std::string norm_str = "bn")
    {
        pconv = PartialConv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(1));

        block->push_back(NormFromString(norm_str, out_channels));
        block->push_back(torch::nn::ReLU());

        block->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).padding(1)));
        block->push_back(NormFromString(norm_str, out_channels));
        block->push_back(torch::nn::ReLU());

        register_module("pconv", pconv);
        register_module("block", block);
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        auto [out_img, out_mask] = pconv->forward(x, mask);
        return {block->forward(out_img), out_mask};
    }


    PartialConv2d pconv = nullptr;
    torch::nn::Sequential block;
};

TORCH_MODULE(PartialBlock);



class GatedBlockImpl : public UnetBlockImpl
{
   public:
    GatedBlockImpl(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int dilation = 1,
                   std::string norm_str = "bn")
    {
        int n_pad_pxl = int(dilation * (kernel_size - 1) / 2);

        feature_transform->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                           .stride(stride)
                                                           .dilation(dilation)
                                                           .padding(n_pad_pxl)));
        feature_transform->push_back(torch::nn::ELU());

        mask_transform->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                        .stride(stride)
                                                        .dilation(dilation)
                                                        .padding(n_pad_pxl)));
        mask_transform->push_back(torch::nn::Sigmoid());

        norm = NormFromString(norm_str, out_channels);

        register_module("feature_transform", feature_transform);
        register_module("mask_transform", mask_transform);
        register_module("norm", norm.ptr());
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        auto x_t = feature_transform->forward(x);
        auto m_t = mask_transform->forward(x);
        auto res = norm.forward(x_t * m_t);
        return {res, mask};
    }

    torch::nn::Sequential feature_transform;
    torch::nn::Sequential mask_transform;
    torch::nn::AnyModule norm;
};

TORCH_MODULE(GatedBlock);

inline torch::nn::AnyModule UnetBlockFromString(const std::string& str, int in_channels, int out_channels,
                                                int kernel_size = 3, int stride = 1, int dilation = 1,
                                                std::string norm_str = "id")
{
    if (str == "basic")
    {
        return torch::nn::AnyModule(BasicBlock(in_channels, out_channels, kernel_size, norm_str));
    }
    else if (str == "gated")
    {
        return torch::nn::AnyModule(GatedBlock(in_channels, out_channels, kernel_size, stride, dilation, norm_str));
    }
    else if (str == "partial")
    {
        return torch::nn::AnyModule(PartialBlock(in_channels, out_channels, kernel_size, norm_str));
    }
    else
    {
        SAIGA_EXIT_ERROR("Unknown activation " + str);
        return {};
    }
}


class DownsampleBlockImpl : public UnetBlockImpl
{
   public:
    using UnetBlockImpl::forward;

    DownsampleBlockImpl(int in_channels, int out_channels, std::string conv_block, std::string norm_str,
                        std::string pooling_str)
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);
        // conv = GatedBlock(in_channels, out_channels, 3, 1, 1, norm_str);
        conv = UnetBlockFromString(conv_block, in_channels, out_channels, 3, 1, 1, norm_str);
        // down = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2, 2}));
        down      = Pooling2DFromString(pooling_str, {2, 2});
        down_mask = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}));

        register_module("conv", conv.ptr());
        register_module("down", down.ptr());
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        auto outputs = down.forward(x);
        if (mask.defined())
        {
            mask = down_mask->forward(mask);
        }
        auto res = conv.forward<std::pair<at::Tensor, at::Tensor>>(outputs, mask);
        return res;
    }

    // GatedBlock conv = nullptr;
    torch::nn::AnyModule conv;
    torch::nn::AnyModule down;
    torch::nn::MaxPool2d down_mask = nullptr;
};

TORCH_MODULE(DownsampleBlock);



class UpsampleBlockImpl : public torch::nn::Module
{
   public:
    UpsampleBlockImpl(int in_channels, int out_channels, std::string upsample_mode = "deconv",
                      std::string norm_str = "id")
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            std::vector<double> scale = {2.0, 2.0};
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
            up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }
        conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);



        register_module("up", up);
        register_module("conv", conv);
    }

    at::Tensor forward(const at::Tensor inputs1, const torch::Tensor inputs2)
    {
        auto in1_up = up->forward(inputs1);

        SAIGA_ASSERT(inputs2.size(2) == in1_up.size(2));
        SAIGA_ASSERT(inputs2.size(3) == in1_up.size(3));

        auto output = conv->forward(torch::cat({in1_up, inputs2}, 1));

        return output.first;
    }

    torch::nn::Sequential up;
    GatedBlock conv = nullptr;
    //    GatedBlock conv;
    //    torch::nn::AvgPool2d down;
};

TORCH_MODULE(UpsampleBlock);


struct MultiScaleUnet2dParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(MultiScaleUnet2dParams);

    virtual void Params(Saiga::SimpleIni& ini_) override
    {
        SAIGA_PARAM_LONG(num_input_layers);
        SAIGA_PARAM_LONG(num_input_channels);
        SAIGA_PARAM_LONG(num_output_channels);
        SAIGA_PARAM_LONG(feature_factor);
        SAIGA_PARAM_LONG(num_layers);
        SAIGA_PARAM_BOOL(add_input_to_filters);
        SAIGA_PARAM_BOOL(channels_last);
        SAIGA_PARAM_BOOL(half_float);

        SAIGA_PARAM_STRING(upsample_mode);
        SAIGA_PARAM_STRING(norm_layer_down);
        SAIGA_PARAM_STRING(norm_layer_up);
        SAIGA_PARAM_STRING(last_act);
        SAIGA_PARAM_STRING(conv_block);
        SAIGA_PARAM_STRING(pooling);
    }


    int num_input_layers        = 5;
    int num_input_channels      = 8;
    int num_output_channels     = 3;
    int feature_factor          = 4;
    int num_layers              = 5;
    bool add_input_to_filters   = false;
    bool channels_last          = false;
    bool half_float             = false;
    std::string upsample_mode   = "bilinear";
    std::string norm_layer_down = "bn";
    std::string norm_layer_up   = "id";
    std::string last_act        = "sigmoid";
    std::string conv_block      = "partial";

    // average, max
    std::string pooling = "average";
};

// Rendering network with UNet architecture and multi-scale input.
//
// Implementation based on the python code from:
// https://github.com/alievk/npbg/blob/master/npbg/models/unet.py
//
//  Args:
//  num_input_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for
//  each input tensor. num_output_channels: Number of output channels. feature_scale: Division factor of number of
//  convolutional channels. The bigger the less parameters in the model. more_layers: Additional down/up-sample layers.
//  upsample_mode: One of 'deconv', 'bilinear' or 'nearest' for ConvTranspose, Bilinear or Nearest upsampling.
//  norm_layer: [unused] One of 'bn', 'in' or 'none' for BatchNorm, InstanceNorm or no normalization. Default: 'bn'.
//  last_act: Last layer activation. One of 'sigmoid', 'tanh' or None.
//  conv_block: Type of convolutional block, like Convolution-Normalization-Activation. One of 'basic', 'partial' or
//  'gated'.
//
class MultiScaleUnet2dImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::vector<int> num_input_channels_per_layer;
        std::vector<int> filters = {4, 8, 16, 32, 64};

        std::vector<int> num_input_channels(params.num_input_layers, params.num_input_channels);
        for (int i = params.num_input_layers; i < 5; ++i)
        {
            num_input_channels.push_back(0);
        }
        for (int i = 0; i < 5; ++i)
        {
            auto& f = filters[i];
            f       = f * params.feature_factor;
            if (params.add_input_to_filters && i >= 1)
            {
                f += num_input_channels[i];
            }
        }


        SAIGA_ASSERT(num_input_channels.size() == filters.size());

        // start = GatedBlock(num_input_channels[0], filters[0]);

        start = UnetBlockFromString(params.conv_block, num_input_channels[0], filters[0], 3, 1, 1, "id");

        down1 = DownsampleBlock(filters[0], filters[1] - num_input_channels[1], params.conv_block,
                                params.norm_layer_down, params.pooling);
        down2 = DownsampleBlock(filters[1], filters[2] - num_input_channels[2], params.conv_block,
                                params.norm_layer_down, params.pooling);

        down3 = DownsampleBlock(filters[2], filters[3] - num_input_channels[3], params.conv_block,
                                params.norm_layer_down, params.pooling);

        if (params.num_layers >= 5)
        {
            down4 = DownsampleBlock(filters[3], filters[4] - num_input_channels[4], params.conv_block,
                                    params.norm_layer_down, params.pooling);
            up4   = UpsampleBlock(filters[4], filters[3], params.upsample_mode, params.norm_layer_up);
        }
        up3 = UpsampleBlock(filters[3], filters[2], params.upsample_mode, params.norm_layer_up);
        up2 = UpsampleBlock(filters[2], filters[1], params.upsample_mode, params.norm_layer_up);
        up1 = UpsampleBlock(filters[1], filters[0], params.upsample_mode, params.norm_layer_up);

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));

        register_module("start", start.ptr());

        register_module("down1", down1);
        register_module("down2", down2);
        register_module("down3", down3);

        if (params.num_layers >= 5)
        {
            register_module("down4", down4);
            register_module("up4", up4);
        }

        register_module("up3", up3);
        register_module("up2", up2);
        register_module("up1", up1);

        register_module("final", final);

        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers);
        SAIGA_ASSERT(masks.size() == params.num_input_layers);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
        }

        std::pair<torch::Tensor, torch::Tensor> d0, d1, d2, d3, d4;
        torch::Tensor u0, u1, u2, u3, u4;

        d0 = start.forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[0], masks[0]);

        d1 = down1->forward(d0);
        if (params.num_input_layers >= 2) d1.first = torch::cat({d1.first, inputs[1]}, 1);

        d2 = down2->forward(d1);
        if (params.num_input_layers >= 3) d2.first = torch::cat({d2.first, inputs[2]}, 1);

        d3 = down3->forward(d2);
        if (params.num_input_layers >= 4) d3.first = torch::cat({d3.first, inputs[3]}, 1);


        if (params.num_layers >= 5)
        {
            d4 = down4->forward(d3);
            if (params.num_input_layers >= 5) d4.first = torch::cat({d4.first, inputs[4]}, 1);
            u4 = d4.first;
            u3 = up4->forward(u4, d3.first);
        }

        u2 = up3->forward(u3, d2.first);
        u1 = up2->forward(u2, d1.first);
        u0 = up1->forward(u1, d0.first);

#if 0
        TensorToImage<unsigned char>(masks[0]).save("mask_in.png");
        TensorToImage<unsigned char>(d0.second).save("mask0.png");
        TensorToImage<unsigned char>(d1.second).save("mask1.png");
        TensorToImage<unsigned char>(d2.second).save("mask2.png");
        TensorToImage<unsigned char>(d3.second).save("mask3.png");
        TensorToImage<unsigned char>(d4.second).save("mask4.png");
        exit(0);
#endif


        return final->forward(u0);
    }

    MultiScaleUnet2dParams params;

    torch::nn::AnyModule start;

    DownsampleBlock down1 = nullptr;
    DownsampleBlock down2 = nullptr;
    DownsampleBlock down3 = nullptr;
    DownsampleBlock down4 = nullptr;

    UpsampleBlock up1 = nullptr;
    UpsampleBlock up2 = nullptr;
    UpsampleBlock up3 = nullptr;
    UpsampleBlock up4 = nullptr;

    torch::nn::Sequential final;
};

TORCH_MODULE(MultiScaleUnet2d);

}  // namespace Saiga
