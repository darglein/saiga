/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "TorchHelper.h"

#include <torch/script.h>

#if __has_include(<torchvision/csrc/models/vgg.h>)
// Use this if torchvision was added as a submodule
#include <torchvision/csrc/models/vgg.h>
#else
// System path otherwise
#include <torchvision/models/vgg.h>
#endif

namespace Saiga
{
// This is a helper class to get the pytorch pretrained vgg loss into c++.
// First, run the following code in python to extract the vgg weights:
//      model =  torchvision.models.vgg19(pretrained=True).features
//      torch.jit.save(torch.jit.script(model), 'vgg_script.pth')
// After that you can create the vgg loss object in c++ and load the weights with:
//      PretrainedVGG19Loss loss("vgg_script.pth");
//
// After that, this loss can be used in regular python code :)
class PretrainedVGG19LossImpl : public torch::nn::Module
{
   public:
    PretrainedVGG19LossImpl(const std::string& file, bool use_average_pool = true, bool from_pytorch = true)
    {
        torch::nn::Sequential features = vision::models::VGG19()->features;

        if (use_average_pool)
        {
            for (auto m : *features)
            {
                auto mod = m.ptr();

                if (auto func_ptr = std::dynamic_pointer_cast<torch::nn::FunctionalImpl>(mod))
                {
                    auto x = torch::zeros({1, 1, 4, 4});
                    x      = func_ptr->forward(x);
                    if (x.size(2) == 2)
                    {
                        seq->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2})));
                    }
                    else
                    {
                        seq->push_back(m);
                    }
                }
                else
                {
                    seq->push_back(m);
                }
            }
        }
        else
        {
            seq = features;
        }

        if (from_pytorch)
        {
            {
                float array[] = {0.485, 0.456, 0.406};
                auto options  = torch::TensorOptions().dtype(torch::kFloat32);
                mean_         = torch::from_blob(array, {1, 3, 1, 1}, options).clone();
            }

            {
                float array[] = {0.229, 0.224, 0.225};
                auto options  = torch::TensorOptions().dtype(torch::kFloat32);
                std_          = torch::from_blob(array, {1, 3, 1, 1}, options).clone();
            }
        }
        else
        {
            {
                float array[] = {103.939 / 255.f, 116.779 / 255.f, 123.680 / 255.f};
                auto options  = torch::TensorOptions().dtype(torch::kFloat32);
                mean_         = torch::from_blob(array, {1, 3, 1, 1}, options).clone();
            }

            {
                float array[] = {1. / 255, 1. / 255, 1. / 255};
                auto options  = torch::TensorOptions().dtype(torch::kFloat32);
                std_          = torch::from_blob(array, {1, 3, 1, 1}, options).clone();
            }
        }

        std::cout << "num layers in vgg " << seq->size() << std::endl;

        if (0)
        {
            layers     = {1, 3, 6, 8, 11, 13, 15};
            last_layer = 15;
        }
        else
        {
            layers     = {1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29};
            last_layer = 29;
        }

        torch::nn::Sequential cpy;

        int i = 0;
        for (auto l : *seq)
        {
            cpy->push_back(l);
            if (i >= last_layer) break;
            i++;
        }
        seq = cpy;
        seq->eval();
        SAIGA_ASSERT(seq->size() == last_layer + 1);


        LoadFromPythonExport(file);

        register_module("features", seq);
        register_buffer("mean", mean_);
        register_buffer("std", std_);
    }

    void LoadFromPythonExport(std::string file)
    {
        torch::jit::Module py_model = torch::jit::load(file);
        auto params                 = py_model.named_parameters();
        std::map<std::string, torch::Tensor> param_map;
        for (auto p : params)
        {
            param_map[p.name] = p.value;
        }

        torch::nn::Module module = *seq;
        for (int i = 0; i < module.children().size(); ++i)
        {
            if (auto conv_ptr = module.children()[i]->as<torch::nn::Conv2d>())
            {
                auto weight = param_map[std::to_string(i) + ".weight"];
                auto bias   = param_map[std::to_string(i) + ".bias"];
                SAIGA_ASSERT(conv_ptr->weight.sizes() == weight.sizes());
                SAIGA_ASSERT(conv_ptr->bias.sizes() == bias.sizes());
                {
                    torch::NoGradGuard ng;
                    conv_ptr->weight.set_(weight);
                    conv_ptr->bias.set_(bias);
                }
            }
        }
    }

    torch::Tensor normalize_inputs(torch::Tensor x) { return (x - mean_) / std_; }

    torch::Tensor forward(torch::Tensor input, torch::Tensor target)
    {
        torch::Tensor features_input  = normalize_inputs(input);
        torch::Tensor features_target = normalize_inputs(target);


        torch::Tensor loss = torch::zeros({1}, torch::TensorOptions().device(input.device()));

        int i = 0;
        for (auto m : *seq)
        {
            features_input  = m.any_forward(features_input).get<torch::Tensor>();
            features_target = m.any_forward(features_target).get<torch::Tensor>();

            if (0)
            {
                auto mod = m.get<torch::nn::Conv2d>();

                float f = features_input[0][0][0][0].item().toFloat();

                std::cout << "f " << f << std::endl;

                PrintTensorInfo(features_input);
                PrintTensorInfo(features_target);
                PrintTensorInfo(mod->weight);
                PrintTensorInfo(mod->bias);

                //                std::cout << mod->weight.mean().item().to<float>() << " " <<
                //                mod->bias.mean().item().to<float>()
                //                          << std::endl;
                //                std::cout << "input,target mean: " << features_input.mean().item().to<float>() << " "
                //                          << features_target.mean().item().to<float>() << std::endl;
                exit(0);
            }

            if (layers.count(i) > 0)
            {
                loss = loss + torch::nn::functional::l1_loss(features_input, features_target);
            }

            if (i >= last_layer)
            {
                break;
            }
            i++;
        }

        return loss;
    }

    int last_layer;
    torch::nn::Sequential seq;

    std::set<int> layers;
    torch::Tensor mean_, std_;
};

TORCH_MODULE(PretrainedVGG19Loss);

}  // namespace Saiga
