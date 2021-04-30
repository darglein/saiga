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
class SAIGA_OPENGL_API Bloom
{
   public:
    Bloom();

    void Render(Texture* hdr_texture, float current_exposure);

    void imgui();

   private:
    bool use_blur = false;
    void resize(int w, int h);
    int w = 0, h = 0;
    int levels = 5;
    std::vector<std::shared_ptr<Texture>> bright_textures;
    std::vector<std::shared_ptr<Texture>> blur_textures;

    std::shared_ptr<Shader> extract_shader, downsample_shader,upsample_shader, blurx_shader, blury_shader, combine_shader;
};

}  // namespace Saiga
