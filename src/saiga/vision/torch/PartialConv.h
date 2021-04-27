/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "TorchHelper.h"


namespace Saiga
{
// C++ Torch Implementation for the Partial Convolution
// Based on the paper:
//      "Image Inpainting for Irregular Holes Using Partial Convolutions"
//      by Guilin Liu et. al.
//
// the implementation is based on the python code from here:
// https://github.com/alievk/npbg/blob/master/npbg/models/conv.py
//
class PartialConv2dImpl : public torch::nn::Conv2dImpl
{
   public:
    // Same parameters as conv2d
    PartialConv2dImpl(torch::nn::Conv2dOptions options) : torch::nn::Conv2dImpl(options)
    {
        weight_maskUpdater = torch::ones({1, 1, options.kernel_size()->at(0), options.kernel_size()->at(1)});
        slide_winsize      = weight_maskUpdater.size(0) * weight_maskUpdater.size(1) * weight_maskUpdater.size(2) *
                        weight_maskUpdater.size(3);
        register_buffer("weight_maskUpdater", weight_maskUpdater);
    }

    torch::Tensor forward(torch::Tensor input, torch::Tensor mask)
    {
        torch::Tensor mask_ratio, update_mask;
        {
            torch::NoGradGuard nograd;

            update_mask =
                torch::nn::functional::detail::conv2d(mask, weight_maskUpdater, {}, options.stride(), options.padding(),
                                                      options.dilation(), options.groups());

            mask_ratio  = slide_winsize / (update_mask + 1e-8);
            update_mask = torch::clamp(update_mask, 0, 1);

            mask_ratio = torch::mul(mask_ratio, update_mask);
        }


        auto raw_out = torch::nn::Conv2dImpl::forward(torch::mul(input, mask));

        torch::Tensor output;
        if (this->bias.defined())
        {
            auto bias_view = this->bias.view({1, options.out_channels(), 1, 1});
            output         = torch::mul(raw_out - bias_view, mask_ratio) + bias_view;
            output         = torch::mul(output, update_mask);
        }
        else
        {
            output = torch::mul(raw_out, mask_ratio);
        }

        return output;
    }

    torch::Tensor weight_maskUpdater;
    float slide_winsize;
};

TORCH_MODULE(PartialConv2d);

}  // namespace Saiga
