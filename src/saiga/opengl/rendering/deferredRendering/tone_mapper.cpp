/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "tone_mapper.h"

#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/shader/shaderLoader.h"


namespace Saiga
{
ToneMapper::ToneMapper()
{
    shader = shaderLoader.load<Shader>("tone_map.glsl");
    uniforms.create(ArrayView<TonemapParameters>(params), GL_STATIC_DRAW);

    camera_response.normalize(1);
}
void ToneMapper::Map(Texture* input_hdr_color_image, Texture* output_ldr_color_image)
{
    if (params_dirty)
    {
        uniforms.update(ArrayView<TonemapParameters>(params));
        response_texture = std::make_shared<Texture1D>();
        response_texture->create(camera_response.irradiance.size(), GL_RED, GL_R32F, GL_FLOAT,
                                 camera_response.irradiance.data());
        params_dirty = false;
    }

    shader->bind();
    input_hdr_color_image->bindImageTexture(0, GL_READ_ONLY);
    output_ldr_color_image->bindImageTexture(1, GL_WRITE_ONLY);
    // response_texture->bindImageTexture(2, GL_READ_ONLY);
    shader->upload(2, response_texture.get(), 0);
    uniforms.bind(3);
    int gw = iDivUp(input_hdr_color_image->getWidth(), 16);
    int gh = iDivUp(input_hdr_color_image->getHeight(), 16);
    shader->dispatchCompute(uvec3(gw, gh, 1));
    shader->unbind();
}
void ToneMapper::imgui()
{
    params_dirty |= ImGui::SliderFloat("exposure", &params.exposure, 0.1, 5);
    params_dirty |= ImGui::SliderFloat3("vignette_coeffs", params.vignette_coeffs.data(), -3, 1);
    params_dirty |= ImGui::SliderFloat2("vignette_offset", params.vignette_offset.data(), -1, 1);

    if (ImGui::Button("gamma response"))
    {
        float gamma = 1.0 / 1.56;
        for (int i = 0; i < camera_response.irradiance.size(); ++i)
        {
            float alpha = float(i) / (camera_response.irradiance.size() - 1);
            if (alpha == 0)
            {
                camera_response.irradiance[i] = 0;
            }
            else
            {
                camera_response.irradiance[i] = pow(alpha, gamma);
            }
        }
        params_dirty = true;
    }


    ImGui::Text("Camera Response");
    ImGui::PlotLines("###response", camera_response.irradiance.data(), camera_response.irradiance.size(), 0, "", 0, 1,
                     ImVec2(100, 80));
}
}  // namespace Saiga
