/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/all.h"
#include "saiga/opengl/opengl.h"

namespace Saiga
{
struct BloomParameters
{
    float bloom_threshold = 1;
    float bloom_strength  = 2;
    int levels            = 4;
    int flags             = 0;

    BloomParameters() { static_assert(sizeof(BloomParameters) == sizeof(float) * 4); }
};


class SAIGA_OPENGL_API Bloom
{
   public:
    enum class DebugMode : int
    {
        NO_DEBUG = 0,
        DEBUG_EXTRACT,
        DEBUG_ADD,
        DEBUG_LAST,
    };

    Bloom(GLenum input_type = GL_RGBA16F);
    void Render(Texture* hdr_texture);
    void imgui();

    bool params_dirty = true;
    BloomParameters params;
    DebugMode mode = DebugMode::NO_DEBUG;
    int extract_downsample = 2;
   private:
    bool use_blur = true;
    void resize(int w, int h);

    void Blur(Texture* source, Texture* target, Texture* tmp);

    int w = 0, h = 0;
    TemplatedUniformBuffer<BloomParameters> uniforms;
    std::vector<std::shared_ptr<Texture>> bright_textures;
    std::vector<std::shared_ptr<Texture>> blur_textures;
    std::shared_ptr<Shader> extract_shader, downsample_shader, upsample_shader, blurx_shader, blury_shader,
        combine_shader, copy_image_shader;
    GLenum input_type;
};

}  // namespace Saiga
