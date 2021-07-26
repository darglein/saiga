/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "bloom.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/opengl/glImageFormat.h"
#include "saiga/opengl/imgui/imgui_opengl.h"

namespace Saiga
{
Bloom::Bloom(GLenum input_type) : input_type(input_type)
{
    auto input_type_str = BindlessImageTypeName(input_type);
    ShaderPart::ShaderCodeInjections injection;
    injection.emplace_back(GL_COMPUTE_SHADER, "#define INPUT_TYPE " + input_type_str, 1);

    extract_shader    = shaderLoader.load<Shader>("compute/bloom_extract_bright.glsl", injection);
    downsample_shader = shaderLoader.load<Shader>("compute/bloom_downsample.glsl", injection);
    upsample_shader   = shaderLoader.load<Shader>("compute/bloom_upsample.glsl", injection);
    combine_shader    = shaderLoader.load<Shader>("compute/bloom_combine_simple.glsl", injection);
    copy_image_shader = shaderLoader.load<Shader>("compute/copy_image.glsl", injection);

    {
        ShaderPart::ShaderCodeInjections sci = injection;
        sci.emplace_back(GL_COMPUTE_SHADER, "#define BLUR_X 1", 3);
        sci.emplace_back(GL_COMPUTE_SHADER, "#define BLUR_SIZE 2", 3);
        blurx_shader = shaderLoader.load<Shader>("compute/compute_blur.glsl", sci);
    }
    {
        ShaderPart::ShaderCodeInjections sci = injection;
        sci.emplace_back(GL_COMPUTE_SHADER, "#define BLUR_Y 1", 3);
        sci.emplace_back(GL_COMPUTE_SHADER, "#define BLUR_SIZE 2", 3);
        blury_shader = shaderLoader.load<Shader>("compute/compute_blur.glsl", sci);
    }

    uniforms.create(params, GL_DYNAMIC_DRAW);
}
void Bloom::imgui()
{
    if (ImGui::CollapsingHeader("Bloom"))
    {
        ImGui::Text("Bloom");
        params_dirty |= ImGui::Checkbox("use_blur", &use_blur);
        params_dirty |= ImGui::SliderFloat("bloom_strength", &params.bloom_strength, 0, 100);
        params_dirty |= ImGui::SliderFloat("bloom_threshold", &params.bloom_threshold, 0, 4);
        params_dirty |= ImGui::SliderInt("levels", &params.levels, 1, 16);
        if (ImGui::SliderInt("extract_downsample", &extract_downsample, 0, 5))
        {
            // force rebuild
            params_dirty = true;
            w            = 0;
        }

        static std::vector<std::string> debug_strs = {"NO_DEBUG", "DEBUG_EXTRACT", "DEBUG_ADD", "DEBUG_LAST"};
        ImGui::Combo("Mode", (int*)&mode, debug_strs);
    }
}
void Bloom::Render(Texture* hdr_texture)
{
    if (hdr_texture->getWidth() != w || hdr_texture->getHeight() != h)
    {
        resize(hdr_texture->getWidth(), hdr_texture->getHeight());
    }

    if (params_dirty)
    {
        resize(w, h);
        uniforms.update(params);
        params_dirty = false;
    }

    uniforms.bind(3);

    if (extract_shader->bind())
    {
        extract_shader->upload(5, *hdr_texture, 0);
        bright_textures.front()->bindImageTexture(1, GL_WRITE_ONLY);
        extract_shader->dispatchComputeImage(bright_textures.front().get(), 16);
        extract_shader->unbind();
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    }

    if (mode == DebugMode::DEBUG_EXTRACT)
    {
        if (copy_image_shader->bind())
        {
            bright_textures.front()->bindImageTexture(0, GL_READ_ONLY);
            hdr_texture->bindImageTexture(1, GL_WRITE_ONLY);
            copy_image_shader->dispatchComputeImage(hdr_texture, 16);
            copy_image_shader->unbind();
        }
        return;
    }

    if (downsample_shader->bind())
    {
        for (int i = 1; i < params.levels; ++i)
        {
            // bright_textures[i - 1]->bindImageTexture(0, GL_READ_ONLY);
            downsample_shader->upload(5, *bright_textures[i - 1], 0);
            bright_textures[i]->bindImageTexture(1, GL_WRITE_ONLY);
            downsample_shader->dispatchComputeImage(bright_textures[i].get(), 16);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        }
        downsample_shader->unbind();
    }

    if (use_blur)
    {
        {
            // blur lowest level
            int i = params.levels - 1;
            if (blurx_shader->bind())
            {
                blurx_shader->upload(5, bright_textures[i], 0);
                blur_textures[i]->bindImageTexture(1, GL_WRITE_ONLY);
                blurx_shader->dispatchComputeImage(bright_textures[i].get(), 16);
                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
                blurx_shader->unbind();
            }

            if (blury_shader->bind())
            {
                blury_shader->upload(5, blur_textures[i], 0);
                bright_textures[i]->bindImageTexture(1, GL_WRITE_ONLY);
                blury_shader->dispatchComputeImage(bright_textures[i].get(), 16);
                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
                blury_shader->unbind();
            }
        }

        for (int i = params.levels - 1; i > 0; i--)
        {
            if (upsample_shader->bind())
            {
                upsample_shader->upload(5, bright_textures[i], 0);
                upsample_shader->upload(7, i);
                upsample_shader->upload(8, params.levels);
                bright_textures[i - 1]->bindImageTexture(1, GL_READ_WRITE);
                upsample_shader->dispatchComputeImage(bright_textures[i - 1].get(), 16);
                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
                upsample_shader->unbind();
            }

            // Don't blur the highest resolution layer to save some time
            if (i > 0)
            {
                if (blurx_shader->bind())
                {
                    blurx_shader->upload(5, bright_textures[i - 1], 0);
                    blur_textures[i - 1]->bindImageTexture(1, GL_WRITE_ONLY);
                    blurx_shader->dispatchComputeImage(bright_textures[i - 1].get(), 16);
                    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
                    blurx_shader->unbind();
                }

                if (blury_shader->bind())
                {
                    blury_shader->upload(5, blur_textures[i - 1], 0);
                    bright_textures[i - 1]->bindImageTexture(1, GL_WRITE_ONLY);
                    blury_shader->dispatchComputeImage(bright_textures[i - 1].get(), 16);
                    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
                    blury_shader->unbind();
                }
            }
        }
    }
    else
    {
        if(upsample_shader->bind())
        {
            for (int i = params.levels - 1; i > 0; i--)
            {
                // bright_textures[i - 1]->bindImageTexture(0, GL_READ_ONLY);
                upsample_shader->upload(5, bright_textures[i], 0);
                upsample_shader->upload(7, i);
                upsample_shader->upload(8, params.levels);
                bright_textures[i - 1]->bindImageTexture(1, GL_READ_WRITE);
                upsample_shader->dispatchComputeImage(bright_textures[i - 1].get(), 16);
                glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
            }
            upsample_shader->unbind();
        }
    }


    if (mode == DebugMode::DEBUG_ADD)
    {
        if (copy_image_shader->bind())
        {
            bright_textures.front()->bindImageTexture(0, GL_READ_ONLY);
            hdr_texture->bindImageTexture(1, GL_WRITE_ONLY);
            copy_image_shader->dispatchComputeImage(hdr_texture, 16);
            copy_image_shader->unbind();
        }
        return;
    }

    if (mode == DebugMode::DEBUG_LAST)
    {
        if (copy_image_shader->bind())
        {
            bright_textures.back()->bindImageTexture(0, GL_READ_ONLY);
            hdr_texture->bindImageTexture(1, GL_WRITE_ONLY);
            copy_image_shader->dispatchComputeImage(hdr_texture, 16);
            copy_image_shader->unbind();
        }
        return;
    }


    if (combine_shader->bind())
    {
        hdr_texture->bindImageTexture(0, GL_READ_WRITE);
        for (int i = 0; i < 1; ++i)
        {
            combine_shader->upload(i, bright_textures[i], i);
        }
        combine_shader->dispatchComputeImage(hdr_texture, 16);
        combine_shader->unbind();
    }
}
void Bloom::resize(int w, int h)
{
    if (w == this->w && h == this->h && blur_textures.size() == params.levels)
    {
        return;
    }

    console << "Resize bloom to " << w << "x" << h << std::endl;
    this->w = w;
    this->h = h;

    bright_textures.clear();
    bright_textures.resize(params.levels);


    blur_textures.clear();
    blur_textures.resize(params.levels);

    for (int i = 0; i < extract_downsample; ++i)
    {
        w /= 2;
        h /= 2;
    }

    for (int l = 0; l < params.levels; ++l)
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
            params.levels = l + 1;
            bright_textures.resize(params.levels);
            blur_textures.resize(params.levels);
            console << "Bloom clamped levels to " << params.levels << std::endl;
            break;
        }
    }
}
void Bloom::Blur(Texture* source, Texture* target, Texture* tmp)
{
    if (blurx_shader->bind())
    {
        blurx_shader->upload(5, source, 0);
        tmp->bindImageTexture(1, GL_WRITE_ONLY);
        blurx_shader->dispatchComputeImage(tmp, 16);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        blurx_shader->unbind();
    }
    if (blury_shader->bind())
    {
        blury_shader->upload(5, tmp, 0);
        target->bindImageTexture(1, GL_WRITE_ONLY);
        blury_shader->dispatchComputeImage(target, 16);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        blury_shader->unbind();
    }
}
}  // namespace Saiga
