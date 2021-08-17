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
        block1->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(1)));
        block1->push_back(NormFromString(norm_str, out_channels));
        block1->push_back(torch::nn::ReLU());

        block2->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).padding(1)));
        block2->push_back(NormFromString(norm_str, out_channels));
        block2->push_back(torch::nn::ReLU());

        register_module("block1", block1);
        register_module("block2", block2);
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        std::pair<at::Tensor, at::Tensor> result;
        result.first  = block1->forward(x);
        result.first  = block2->forward(result.first);
        result.second = mask;
        return result;
    }

    torch::nn::Sequential block1;
    torch::nn::Sequential block2;
};

TORCH_MODULE(BasicBlock);


class PartialBlockImpl : public UnetBlockImpl
{
   public:
    PartialBlockImpl(int in_channels, int out_channels, int kernel_size = 3, bool multi_channel = false,
                     std::string norm_str = "bn")
    {
        pconv1 =
            PartialConv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(1), multi_channel);
        block1->push_back(NormFromString(norm_str, out_channels));
        block1->push_back(torch::nn::ReLU());

        pconv2 =
            PartialConv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).padding(1), multi_channel);
        block2->push_back(NormFromString(norm_str, out_channels));
        block2->push_back(torch::nn::ReLU());

        register_module("pconv1", pconv1);
        register_module("pconv2", pconv2);
        register_module("block1", block1);
        register_module("block2", block2);
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        std::pair<at::Tensor, at::Tensor> result;
        result       = pconv1->forward(x, mask);
        result.first = block1->forward(result.first);

        result       = pconv2->forward(result.first, result.second);
        result.first = block2->forward(result.first);
        return result;
    }


    PartialConv2d pconv1 = nullptr;
    PartialConv2d pconv2 = nullptr;
    torch::nn::Sequential block1, block2;
};

TORCH_MODULE(PartialBlock);



class GatedBlockImpl : public UnetBlockImpl
{
   public:
    GatedBlockImpl(int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int dilation = 1,
                   std::string norm_str = "bn", std::string activation_str = "elu")
    {
        int n_pad_pxl = int(dilation * (kernel_size - 1) / 2);

        SAIGA_ASSERT(kernel_size == 3);
        SAIGA_ASSERT(n_pad_pxl == 1);


        feature_transform->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                           .stride(stride)
                                                           .dilation(dilation)
                                                           .padding(n_pad_pxl)));
        feature_transform->push_back(ActivationFromString(activation_str));

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
        return torch::nn::AnyModule(PartialBlock(in_channels, out_channels, kernel_size, false, norm_str));
    }
    else if (str == "partial_multi")
    {
        return torch::nn::AnyModule(PartialBlock(in_channels, out_channels, kernel_size, true, norm_str));
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
        conv = UnetBlockFromString(conv_block, in_channels, out_channels, 3, 1, 1, norm_str);

        if (pooling_str == "conv")
        {
            down = torch::nn::AnyModule(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 2).stride(2).padding(0)));
            down_mask = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}));
        }
        else if (pooling_str == "partial_multi")
        {
            down_combined = torch::nn::AnyModule(
                PartialConv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 2).stride(2).padding(0), true));
        }
        else
        {
            down      = Pooling2DFromString(pooling_str, {2, 2});
            down_mask = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}));
        }

        register_module("conv", conv.ptr());

        if (down_combined.is_empty())
        {
            register_module("down", down.ptr());
            register_module("down_mask", down_mask.ptr());
        }
        else
        {
            register_module("down_combined", down_combined.ptr());
        }
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        std::pair<at::Tensor, at::Tensor> downsampled_input;

        if (down_combined.is_empty())
        {
            downsampled_input.first = down.forward(x);
            if (mask.defined())
            {
                downsampled_input.second = down_mask->forward(mask);
            }
        }
        else
        {
            downsampled_input = down_combined.forward<std::pair<at::Tensor, at::Tensor>>(x, mask);
        }
        auto res = conv.forward<std::pair<at::Tensor, at::Tensor>>(downsampled_input.first, downsampled_input.second);
        return res;
    }

    // GatedBlock conv = nullptr;
    torch::nn::AnyModule conv;

    torch::nn::AnyModule down_combined;
    torch::nn::AnyModule down;
    torch::nn::MaxPool2d down_mask = nullptr;
};

TORCH_MODULE(DownsampleBlock);



class UpsampleBlockImpl : public torch::nn::Module
{
   public:
    UpsampleBlockImpl(int in_channels, int out_channels, std::string conv_block, std::string upsample_mode = "deconv",
                      std::string norm_str = "id")
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        if (upsample_mode != "deconv")
        {
            if (conv_block == "partial_multi")
            {
                conv1 = torch::nn::AnyModule(
                    PartialConv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1), true));
            }
            else
            {
                conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
            }
        }
        // conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);
        conv2 = UnetBlockFromString(conv_block, out_channels * 2, out_channels, 3, 1, 1, norm_str);


        up_mask = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest));

        register_module("up", up);
        register_module("up_mask", up_mask);

        if (!conv1.is_empty())
        {
            register_module("conv1", conv1.ptr());
        }
        register_module("conv2", conv2.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        // SAIGA_ASSERT(skip.first.size(2) == same_layer_as_skip.first.size(2) &&
        //              skip.first.size(3) == same_layer_as_skip.first.size(3));
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
        {
            return torch::cat({below, skip}, 1);
        }
        else
        {
            return torch::cat({below, CenterCrop2D(skip, below.sizes())}, 1);
        }
    }

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> skip)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(skip.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> same_layer_as_skip;
        same_layer_as_skip.first = up->forward(layer_below.first);

        if (layer_below.second.defined())
        {
            same_layer_as_skip.second = up_mask->forward(layer_below.second);
            // SAIGA_ASSERT(skip.second.size(2) == same_layer_as_skip.second.size(2) &&
            //              skip.second.size(3) == same_layer_as_skip.second.size(3));
        }

        if (!conv1.is_empty())
        {
            same_layer_as_skip =
                conv1.forward<std::pair<at::Tensor, at::Tensor>>(same_layer_as_skip.first, same_layer_as_skip.second);
        }


        std::pair<at::Tensor, at::Tensor> output;
        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        output.first = CombineBridge(same_layer_as_skip.first, skip.first);

        if (layer_below.second.defined())
        {
            // output.second = torch::cat({same_layer_as_skip.second, skip.second}, 1);
            output.second = CombineBridge(same_layer_as_skip.second, skip.second);
        }

        return conv2.forward<std::pair<at::Tensor, at::Tensor>>(output.first, output.second);
    }

    torch::nn::Sequential up;
    torch::nn::Upsample up_mask = nullptr;
    torch::nn::AnyModule conv1;
    torch::nn::AnyModule conv2;
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
        SAIGA_PARAM_STRING(conv_block_up);
        SAIGA_PARAM_STRING(pooling);
    }

    static constexpr int max_layers = 5;

    int num_input_layers        = 5;
    int num_input_channels      = 8;
    int num_output_channels     = 3;
    int feature_factor          = 4;
    int num_layers              = 5;
    bool add_input_to_filters   = false;
    bool channels_last          = false;
    bool half_float             = false;
    std::string upsample_mode   = "bilinear";
    std::string norm_layer_down = "id";
    std::string norm_layer_up   = "id";
    std::string last_act        = "";
    std::string conv_block      = "gated";
    std::string conv_block_up   = "gated";

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

            if (i >= 1)
            {
                SAIGA_ASSERT(f >= num_input_channels[0]);
            }
        }


        SAIGA_ASSERT(num_input_channels.size() == filters.size());

        start = UnetBlockFromString(params.conv_block, num_input_channels[0], filters[0], 3, 1, 1, "id");
        register_module("start", start.ptr());

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);

        for (int i = 0; i < params.num_layers - 1; ++i)
        {
            // Down[i] transforms from layer (i) -> (i+1)
            int down_in  = filters[i];
            int down_out = filters[i + 1] - num_input_channels[i + 1];
            down[i] = DownsampleBlock(down_in, down_out, params.conv_block, params.norm_layer_down, params.pooling);
            register_module("down" + std::to_string(i + 1), down[i]);

            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i]      = UpsampleBlock(up_in, up_out, params.conv_block_up, params.upsample_mode, params.norm_layer_up);
            register_module("up" + std::to_string(i + 1), up[i]);
        }

        multi_channel_masks = params.conv_block == "partial_multi";
        need_up_masks       = params.conv_block_up == "partial_multi";
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
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
            SAIGA_ASSERT(masks[i].requires_grad() == false);
        }

        if (multi_channel_masks)
        {
            torch::NoGradGuard ngg;
            // multi channel partial convolution needs a mask value for each channel.
            // Here, we just repeat the masks along the channel dimension.
            for (int i = 0; i < inputs.size(); ++i)
            {
                auto& ma = masks[i];
                auto& in = inputs[i];
                if (ma.size(1) == 1 && in.size(1) > 1)
                {
                    ma = ma.repeat({1, in.size(1), 1, 1});
                }
            }
        }

        std::pair<torch::Tensor, torch::Tensor> d[MultiScaleUnet2dParams::max_layers];

        d[0] = start.forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[0], masks[0]);

        // Loops Range: [1,2, ... , layers-1]
        // At 5 layers we have only 4 down-sample stages
        for (int i = 1; i < params.num_layers; ++i)
        {
            d[i] = down[i - 1]->forward(d[i - 1]);
            if (params.num_input_layers > i)
            {
                d[i].first = torch::cat({d[i].first, inputs[i]}, 1);
                if (multi_channel_masks) d[i].second = torch::cat({d[i].second, masks[i]}, 1);
            }
        }

        if (!need_up_masks)
        {
            for (int i = 0; i < params.num_layers; ++i)
            {
                d[i].second = {};
            }
        }

        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 1; i >= 1; --i)
        {
            d[i - 1] = up[i - 1]->forward(d[i], d[i - 1]);
        }
        return final->forward(d[0].first);
    }

    MultiScaleUnet2dParams params;
    bool multi_channel_masks = false;
    bool need_up_masks       = false;

    torch::nn::AnyModule start;
    torch::nn::Sequential final;

    DownsampleBlock down[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
    UpsampleBlock up[MultiScaleUnet2dParams::max_layers - 1]     = {nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2d);

}  // namespace Saiga
