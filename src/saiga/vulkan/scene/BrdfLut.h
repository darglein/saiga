/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"

#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * The split-sum BRDF LUT of S36 (A3; shader/vulkan/scene/lighting/brdf_lut.comp): size x size RG16F, texel (x, y) =
 * the scale A and bias B of F0 for NdotV = (x + 0.5) / size and perceptual roughness (y + 0.5) / size, computed on the
 * GPU (kSamples points of a fixed Hammersley set, deterministic) by the first init of the process and uploaded from a
 * process-wide copy by later ones (the integration costs about 0.1 s on lavapipe per RenderScene). Binding 5 of the
 * lighting set (descriptorInfo: a linear sampler that clamps to the edge texels, so texture(lut, vec2(NdotV,
 * roughness)) interpolates between texel centres and holds the edge values for NdotV or roughness near 0 and 1); read
 * by pbr.glsl's ambient and energy compensation.
 */
class SAIGA_VULKAN_API BrdfLut
{
   public:
    static constexpr int kSize         = 64;
    static constexpr uint32_t kSamples = 2048;

    // Computes (or uploads) the LUT (blocking, on the main queue); the texture is in eShaderReadOnlyOptimal afterwards
    void init(VulkanBase& base);
    void destroy()
    {
        texture.destroy();
        sampler.destroy();
    }

    Texture2D& lut() { return texture; }
    // The binding of the lighting set: the LUT with its clamping sampler
    vk::DescriptorImageInfo descriptorInfo() { return texture.getDescriptorInfo(sampler); }
    // The LUTs computed on the GPU in this process (the others were uploaded)
    static int computations();
    // The LUT read back as floats (A, B per texel, row y = roughness), blocking
    std::vector<vec2> read();

    // The CPU reference of one texel's integral (tests): the same estimator with samples Hammersley points
    static vec2 integrate(float NdotV, float roughness, uint32_t samples);

   private:
    Texture2D texture;
    Sampler sampler;
    void compute(VulkanBase& base);
    void upload(VulkanBase& base, const std::vector<uint8_t>& bytes);
};

}  // namespace Vulkan
}  // namespace Saiga
