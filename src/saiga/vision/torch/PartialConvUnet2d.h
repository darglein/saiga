/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "PartialConv.h"
#include "TorchHelper.h"

namespace Saiga
{
// Check upsample align corner

class BasicBlockImpl : public torch::nn::Module
{
   public:
    BasicBlockImpl(int in_channels, int out_channels, int kernel_size = 3,
                   torch::nn::AnyModule normalization = torch::nn::AnyModule(torch::nn::BatchNorm2d()))
    {
        block->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(1)));
        block->push_back(normalization);
        block->push_back(torch::nn::ReLU());

        block->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).padding(1)));
        block->push_back(normalization);
        block->push_back(torch::nn::ReLU());

        register_module("block", block);
    }

    at::Tensor forward(at::Tensor x) { return block->forward(x); }

    torch::nn::Sequential block;
};

TORCH_MODULE(BasicBlock);


class PartialBlockImpl : public torch::nn::Module
{
   public:
    PartialBlockImpl(int in_channels, int out_channels, int kernel_size = 3,
                     torch::nn::AnyModule normalization = torch::nn::AnyModule(torch::nn::BatchNorm2d()))
    {
        pconv = PartialConv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(1));

        block->push_back(normalization);
        block->push_back(torch::nn::ReLU());

        block->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).padding(1)));
        block->push_back(normalization);
        block->push_back(torch::nn::ReLU());

        register_module("pconv", pconv);
        register_module("block", block);
    }

    at::Tensor forward(at::Tensor x, torch::Tensor mask)
    {
        auto output = pconv->forward(x, mask);
        return block->forward(output);
    }

    PartialConv2d pconv = nullptr;
    torch::nn::Sequential block;
};

TORCH_MODULE(PartialBlock);



class GatedBlockImpl : public torch::nn::Module
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

    at::Tensor forward(at::Tensor x)
    {
        auto x_t = feature_transform->forward(x);
        auto m_t = mask_transform->forward(x);
        return norm.forward(x_t * m_t);
    }
    torch::nn::Sequential feature_transform;
    torch::nn::Sequential mask_transform;
    torch::nn::AnyModule norm;
};

TORCH_MODULE(GatedBlock);



class DownsampleBlockImpl : public torch::nn::Module
{
   public:
    DownsampleBlockImpl(int in_channels, int out_channels)
    {
        conv = GatedBlock(in_channels, out_channels);
        down = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2, 2}));


        register_module("conv", conv);
        register_module("down", down);
    }

    at::Tensor forward(at::Tensor x)
    {
        auto outputs = down->forward(x);
        outputs      = conv->forward(outputs);
        return outputs;
    }

    GatedBlock conv           = nullptr;
    torch::nn::AvgPool2d down = nullptr;
};

TORCH_MODULE(DownsampleBlock);



class UpsampleBlockImpl : public torch::nn::Module
{
   public:
    UpsampleBlockImpl(int out_channels, std::string upsample_mode = "deconv")
    {
        int num_filt = out_channels * 2;

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(num_filt, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            std::vector<double> scale = {2.0, 2.0};
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
            up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filt, out_channels, 3).padding(1)));
        }
        conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, "id");



        register_module("up", up);
        register_module("conv", conv);
    }

    at::Tensor forward(at::Tensor inputs1, torch::Tensor inputs2)
    {
        auto in1_up = up->forward(inputs1);

        SAIGA_ASSERT(inputs2.size(2) == in1_up.size(2));
        SAIGA_ASSERT(inputs2.size(3) == in1_up.size(3));

        auto output = conv->forward(torch::cat({in1_up, inputs2}, 1));
        return output;
    }

    torch::nn::Sequential up;
    GatedBlock conv = nullptr;
    //    GatedBlock conv;
    //    torch::nn::AvgPool2d down;
};

TORCH_MODULE(UpsampleBlock);

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
class PartialConvUnet2dImpl : public torch::nn::Module
{
   public:
    PartialConvUnet2dImpl(std::vector<int> _num_input_channels, int num_output_channels = 3, int feature_scale = 4,
                          int more_layers = 0, std::string upsample_mode = "bilinear", std::string norm_layer = "bn",
                          std::string last_act = "sigmoid", std::string conv_block = "partial")
        : num_input_channels(_num_input_channels)
    {
        std::vector<int> num_input_channels_per_layer;
        std::vector<int> filters = {64, 128, 256, 512, 1024};

        for (auto& f : filters)
        {
            f = f / feature_scale;
        }

        if (num_input_channels.size() < 5)
        {
            for (int i = num_input_channels.size(); i < 5; ++i)
            {
                num_input_channels.push_back(0);
            }
        }


        SAIGA_ASSERT(num_input_channels.size() == filters.size());

        start = GatedBlock(num_input_channels[0], filters[0]);

        down1 = DownsampleBlock(filters[0], filters[1] - num_input_channels[1]);
        down2 = DownsampleBlock(filters[1], filters[2] - num_input_channels[2]);
        down3 = DownsampleBlock(filters[2], filters[3] - num_input_channels[3]);
        down4 = DownsampleBlock(filters[3], filters[4] - num_input_channels[4]);

        up4 = UpsampleBlock(filters[3], upsample_mode);
        up3 = UpsampleBlock(filters[2], upsample_mode);
        up2 = UpsampleBlock(filters[1], upsample_mode);
        up1 = UpsampleBlock(filters[0], upsample_mode);

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], num_output_channels, 1)));
        final->push_back(ActivationFromString(last_act));

        register_module("start", start);

        register_module("down1", down1);
        register_module("down2", down2);
        register_module("down3", down3);
        register_module("down4", down4);


        register_module("up4", up4);
        register_module("up3", up3);
        register_module("up2", up2);
        register_module("up1", up1);

        register_module("final", final);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        // debug check if input has correct format
        for (int i = 0; i < num_input_channels.size(); ++i)
        {
            if (num_input_channels[i] > 0)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(num_input_channels[i] == inputs[i].size(1));
            }
        }


        auto d0 = start->forward(inputs[0]);

        auto d1 = down1->forward(d0);
        if (num_input_channels[1]) d1 = torch::cat({d1, inputs[1]}, 1);

        auto d2 = down2->forward(d1);
        if (num_input_channels[2]) d2 = torch::cat({d2, inputs[2]}, 1);

        auto d3 = down3->forward(d2);
        if (num_input_channels[3]) d3 = torch::cat({d3, inputs[3]}, 1);

        auto d4 = down4->forward(d3);
        if (num_input_channels[4]) d4 = torch::cat({d4, inputs[4]}, 1);

        auto u4 = up4->forward(d4, d3);
        auto u3 = up3->forward(u4, d2);
        auto u2 = up2->forward(u3, d1);
        auto u1 = up1->forward(u2, d0);

        return final->forward(u1);
    }

    std::vector<int> num_input_channels;

    GatedBlock start = nullptr;

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

TORCH_MODULE(PartialConvUnet2d);

}  // namespace Saiga
