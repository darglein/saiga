/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "bloom.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/opengl/imgui/imgui_opengl.h"

namespace Saiga
{
Bloom::Bloom()
{
    extract_shader    = shaderLoader.load<Shader>("compute/bloom_extract_bright.glsl");
    downsample_shader = shaderLoader.load<Shader>("compute/bloom_downsample.glsl");
    upsample_shader   = shaderLoader.load<Shader>("compute/bloom_upsample.glsl");
    combine_shader    = shaderLoader.load<Shader>("compute/bloom_combine_simple.glsl");

    {
        ShaderPart::ShaderCodeInjections sci;
        sci.emplace_back(GL_COMPUTE_SHADER, "#define BLUR_X 1", 3);
        blurx_shader = shaderLoader.load<Shader>("compute/compute_blur.glsl", sci);
    }
    {
        ShaderPart::ShaderCodeInjections sci;
        sci.emplace_back(GL_COMPUTE_SHADER, "#define BLUR_Y 1", 3);
        blury_shader = shaderLoader.load<Shader>("compute/compute_blur.glsl", sci);
    }
}
void Bloom::imgui()
{
    if(ImGui::CollapsingHeader("Bloom"))
    {
        ImGui::Text("Bloom");
        ImGui::Checkbox("use_blur", &use_blur);
        ImGui::SliderFloat("bloom_strength", &bloom_strength, 0, 1);
        if (ImGui::SliderInt("levels", &levels, 1, 16))
        {
            resize(w, h);
        }
    }
}
void Bloom::Render(Texture* hdr_texture, float current_exposure)
{
    if (hdr_texture->getWidth() != w || hdr_texture->getHeight() != h)
    {
        resize(hdr_texture->getWidth(), hdr_texture->getHeight());
    }


    extract_shader->bind();
    hdr_texture->bindImageTexture(0, GL_READ_ONLY);
    bright_textures.front()->bindImageTexture(1, GL_WRITE_ONLY);
    extract_shader->upload(0, current_exposure);
    extract_shader->dispatchComputeImage(hdr_texture, 16);
    extract_shader->unbind();


    downsample_shader->bind();
    for (int i = 1; i < levels; ++i)
    {
        // bright_textures[i - 1]->bindImageTexture(0, GL_READ_ONLY);
        downsample_shader->upload(5, bright_textures[i - 1], 0);
        bright_textures[i]->bindImageTexture(1, GL_WRITE_ONLY);
        downsample_shader->dispatchComputeImage(bright_textures[i].get(), 16);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    }
    downsample_shader->unbind();


    if (use_blur)
    {
        //        for (int i = 0; i < levels; ++i)
        //        {
        //            blurx_shader->bind();
        //            blurx_shader->upload(5, bright_textures[i], 0);
        //            blur_textures[i]->bindImageTexture(1, GL_WRITE_ONLY);
        //            blurx_shader->dispatchComputeImage(bright_textures[i].get(), 16);
        //            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        //            blurx_shader->unbind();
        //
        //            blury_shader->bind();
        //            blury_shader->upload(5, blur_textures[i], 0);
        //            bright_textures[i]->bindImageTexture(1, GL_WRITE_ONLY);
        //            blury_shader->dispatchComputeImage(bright_textures[i].get(), 16);
        //            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        //            blury_shader->unbind();
        //        }

        {
            int i = levels - 1;
            blurx_shader->bind();
            blurx_shader->upload(5, bright_textures[i], 0);
            blur_textures[i]->bindImageTexture(1, GL_WRITE_ONLY);
            blurx_shader->dispatchComputeImage(bright_textures[i].get(), 16);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
            blurx_shader->unbind();

            blury_shader->bind();
            blury_shader->upload(5, blur_textures[i], 0);
            bright_textures[i]->bindImageTexture(1, GL_WRITE_ONLY);
            blury_shader->dispatchComputeImage(bright_textures[i].get(), 16);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
            blury_shader->unbind();
        }

        for (int i = levels - 1; i > 0; i--)
        {
            upsample_shader->bind();
            upsample_shader->upload(5, bright_textures[i], 0);
            upsample_shader->upload(7, float(i) / (levels - 1));
            bright_textures[i - 1]->bindImageTexture(1, GL_READ_WRITE);
            upsample_shader->dispatchComputeImage(bright_textures[i - 1].get(), 16);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
            upsample_shader->unbind();


            if (i >= 2)
            {
                blurx_shader->bind();
                blurx_shader->upload(5, bright_textures[i - 1], 0);
                blur_textures[i - 1]->bindImageTexture(1, GL_WRITE_ONLY);
                blurx_shader->dispatchComputeImage(bright_textures[i - 1].get(), 16);
                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
                blurx_shader->unbind();

                blury_shader->bind();
                blury_shader->upload(5, blur_textures[i - 1], 0);
                bright_textures[i - 1]->bindImageTexture(1, GL_WRITE_ONLY);
                blury_shader->dispatchComputeImage(bright_textures[i - 1].get(), 16);
                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
                blury_shader->unbind();
            }
        }
    }
    else
    {
        upsample_shader->bind();
        for (int i = levels - 1; i > 0; i--)
        {
            // bright_textures[i - 1]->bindImageTexture(0, GL_READ_ONLY);
            upsample_shader->upload(5, bright_textures[i], 0);
            bright_textures[i - 1]->bindImageTexture(1, GL_READ_WRITE);
            upsample_shader->dispatchComputeImage(bright_textures[i - 1].get(), 16);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        }
        upsample_shader->unbind();
    }


    combine_shader->bind();
    hdr_texture->bindImageTexture(0, GL_READ_WRITE);
    for (int i = 0; i < 1; ++i)
    {
        combine_shader->upload(i, bright_textures[i], i);
    }
    combine_shader->upload(7, bloom_strength);
    combine_shader->dispatchComputeImage(hdr_texture, 16);
    combine_shader->unbind();
}
void Bloom::resize(int w, int h)
{
    console << "Resize bloom to " << w << "x" << h << std::endl;
    this->w = w;
    this->h = h;

    bright_textures.clear();
    bright_textures.resize(levels);


    blur_textures.clear();
    blur_textures.resize(levels);

    for (int l = 0; l < levels; ++l)
    {
        bright_textures[l] = std::make_shared<Texture>();
        bright_textures[l]->create(w, h, GL_RGBA, GL_RGBA16F, GL_HALF_FLOAT);

        blur_textures[l] = std::make_shared<Texture>();
        blur_textures[l]->create(w, h, GL_RGBA, GL_RGBA16F, GL_HALF_FLOAT);
        w /= 2;
        h /= 2;


        // stop here
        if (w <= 1 || h <= 1)
        {
            levels = l + 1;
            bright_textures.resize(levels);
            blur_textures.resize(levels);
            console << "Bloom clamped levels to " << levels << std::endl;
            break;
        }
    }
}
}  // namespace Saiga
