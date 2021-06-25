/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "tone_mapper.h"

#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/glImageFormat.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/shader/shaderLoader.h"


namespace Saiga
{
const char* tonemap_operator_names[6] = {"Gamma", "Texture", "Reinhard", "UE3", "UC2", "Drago"};


ToneMapper::ToneMapper(GLenum input_type) : input_type(input_type)
{
    auto input_type_str = BindlessImageTypeName(input_type);
    ShaderPart::ShaderCodeInjections injection;
    injection.emplace_back(GL_COMPUTE_SHADER, "#define INPUT_TYPE " + input_type_str, 1);

    shader        = shaderLoader.load<Shader>("tone_map.glsl", injection);
    shader_linear = shaderLoader.load<Shader>("tone_map_linear.glsl", injection);


    uniforms.create(params, GL_DYNAMIC_DRAW);
    tmp_buffer.create(tmp_params, GL_DYNAMIC_DRAW);

    camera_response.MakeGamma(1.0 / 2.2);
    camera_response.normalize(1);
    params_dirty = true;
}


void ToneMapper::MapLinear(Texture* input_hdr_color_image)
{
    SAIGA_ASSERT(input_hdr_color_image->InternalFormat() == input_type);
    if (params_dirty)
    {
        uint32_t flags = (auto_exposure << 0) | (auto_white_balance << 1);
        params.flags   = flags;

        uniforms.update(params);
        response_texture = std::make_shared<Texture1D>();
        response_texture->create(camera_response.irradiance.size(), GL_RED, GL_R32F, GL_FLOAT,
                                 camera_response.irradiance.data());
        params_dirty = false;
    }

    if (auto_exposure || auto_white_balance)
    {
        ComputeOptimalExposureValue(input_hdr_color_image);
    }

    if (shader_linear->bind())
    {
        input_hdr_color_image->bindImageTexture(0, GL_READ_WRITE);
        uniforms.bind(3);
        tmp_buffer.bind(4);
        int gw = iDivUp(input_hdr_color_image->getWidth(), 16);
        int gh = iDivUp(input_hdr_color_image->getHeight(), 16);
        shader_linear->dispatchCompute(uvec3(gw, gh, 1));
        shader_linear->unbind();
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    }
}


void ToneMapper::Map(Texture* input_hdr_color_image, Texture* output_ldr_color_image)
{
    SAIGA_ASSERT(input_hdr_color_image->InternalFormat() == input_type);
    SAIGA_ASSERT(output_ldr_color_image->InternalFormat() == GL_RGBA8);
    SAIGA_ASSERT(response_texture);

    if (shader->bind())
    {
        input_hdr_color_image->bindImageTexture(0, GL_READ_ONLY);
        output_ldr_color_image->bindImageTexture(1, GL_WRITE_ONLY);
        shader->upload(2, *response_texture, 0);
        shader->upload(3, gamma);
        shader->upload(4, tm_operator);
        int gw = iDivUp(input_hdr_color_image->getWidth(), 16);
        int gh = iDivUp(input_hdr_color_image->getHeight(), 16);
        shader->dispatchCompute(uvec3(gw, gh, 1));
        shader->unbind();
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    }
}

vec3 ColorTemperatureToRGB(float temperatureInKelvins)
{
    vec3 retColor;

    temperatureInKelvins = clamp(temperatureInKelvins, 1000.0, 40000.0) / 100.0;

    if (temperatureInKelvins <= 66.0)
    {
        retColor(0) = 1.0;
        retColor(1) = saturate(0.39008157876901960784 * log(temperatureInKelvins) - 0.63184144378862745098);
    }
    else
    {
        float t     = temperatureInKelvins - 60.0;
        retColor(0) = saturate(1.29293618606274509804 * pow(t, -0.1332047592));
        retColor(1) = saturate(1.12989086089529411765 * pow(t, -0.0755148492));
    }

    if (temperatureInKelvins >= 66.0)
        retColor(2) = 1.0;
    else if (temperatureInKelvins <= 19.0)
        retColor(2) = 0.0;
    else
        retColor(2) = saturate(0.54320678911019607843 * log(temperatureInKelvins - 10.0) - 1.19625408914);

    return retColor;
}


void ToneMapper::imgui()
{
    if (ImGui::CollapsingHeader("tone mapping"))
    {
        params_dirty |= ImGui::Checkbox("auto_exposure", &auto_exposure);
        if (auto_exposure && download_tmp_values)
        {
            ImGui::SameLine();
            ImGui::Text("exp: %f", computed_exposure);
        }
        params_dirty |= ImGui::Checkbox("auto_white_balance", &auto_white_balance);
        if (auto_white_balance && download_tmp_values)
        {
            ImGui::SameLine();

            ImGui::Text("white: %f %f %f", computed_white_point(0), computed_white_point(1), computed_white_point(2));
        }

        params_dirty |= ImGui::Checkbox("download_tmp_values (performance warning)", &download_tmp_values);
        ImGui::Text("Exposure Value (log2 scale)");
        params_dirty |= ImGui::SliderFloat("###exposure_value", &params.exposure_value, -5, 20);
        ImGui::SameLine();
        if (ImGui::Button("reset EV"))
        {
            params.exposure_value = 0;
            params_dirty          = true;
        }

        params_dirty |= ImGui::SliderFloat3("vignette_coeffs", params.vignette_coeffs.data(), -3, 1);
        params_dirty |= ImGui::SliderFloat2("vignette_offset", params.vignette_offset.data(), -1, 1);

        ImGui::Separator();
        ImGui::Text("Camera Response");


        if (ImGui::Combo("Operator", &tm_operator, tonemap_operator_names, 6))
        {
            params_dirty = true;
        }

        if (ImGui::Button("gamma = 2.2"))
        {
            camera_response.MakeGamma(2.2);
            params_dirty = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("gamma = 1"))
        {
            camera_response.MakeGamma(1);
            params_dirty = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("gamma = 1.0/2.2"))
        {
            camera_response.MakeGamma(1.0 / 2.2);
            params_dirty = true;
        }
        static float gamma = 1.0 / 1.5;
        ImGui::SliderFloat("gamma", &gamma, 0, 4);
        if (ImGui::Button("gamma response"))
        {
            camera_response.MakeGamma(gamma);
            params_dirty = true;
        }

        ImGui::PlotLines("###response", camera_response.irradiance.data(), camera_response.irradiance.size(), 0, "", 0,
                         1, ImVec2(100, 80));

        ImGui::Separator();
        ImGui::Text("White Balance");
        if (ImGui::ColorEdit3("white_point", params.white_point.data()))
        {
            params_dirty = true;
        }
        if (ImGui::SliderFloat("color_temperature", &color_temperature, 1000, 15000))
        {
            params.white_point = ColorTemperatureToRGB(color_temperature);
            //        params.white_point = vec3(1, 1, 1).array() / params.white_point.array();
            //        float max_l        = params.white_point.maxCoeff();
            //        params.white_point /= max_l;
            params_dirty = true;
        }
    }
}
void ToneMapper::ComputeOptimalExposureValue(Texture* input_hdr_color_image)
{
    if (!average_brightness_shader)
    {
        average_brightness_shader = shaderLoader.load<Shader>("tone_map_average_brightness.glsl");
    }

    tmp_params = TonemapTempParameters();
    tmp_buffer.update(tmp_params);

    if (average_brightness_shader->bind())
    {
        tmp_buffer.bind(4);
        input_hdr_color_image->bindImageTexture(0, GL_READ_ONLY);
        int gw = iDivUp(input_hdr_color_image->getWidth(), 16);
        int gh = iDivUp(input_hdr_color_image->getHeight(), 16);
        average_brightness_shader->dispatchCompute(uvec3(gw, gh, 1));
        average_brightness_shader->unbind();

        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        if (download_tmp_values)
        {
            // this is every inefficient because we stall the GL pipeline!
            tmp_buffer.get(tmp_params);

            float average_luminace = tmp_params.average_color_luminace.w();
            computed_exposure      = log2(std::max(average_luminace / 0.33, 1e-4));

            vec3 avg             = tmp_params.average_color_luminace.head<3>();
            float alpha          = avg[1] / avg[0];
            float beta           = avg[1] / avg[2];
            computed_white_point = vec3(alpha, 1, beta);
        }
    }
    else
    {
        tmp_params.average_color_luminace = vec4(1, 1, 1, 1);
        tmp_buffer.update(tmp_params);
    }
}

}  // namespace Saiga
