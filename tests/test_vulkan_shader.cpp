/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/Shader/GLSL.h"
#include "saiga/vulkan/Shader/ShaderPipeline.h"

#include "vulkan_test_context.h"

using namespace Saiga;

// The GLSL -> SPIR-V compiler (glslang) is initialized with the VulkanBase of the fixture.
class VulkanShader : public VulkanTest
{
   protected:
    static constexpr uint32_t spirv_magic = 0x07230203;
};

TEST_F(VulkanShader, CompileFromString)
{
    std::string source = R"(#version 450
layout(location = 0) out vec4 outColor;
void main()
{
    outColor = vec4(1, 0, 0, 1);
}
)";
    auto spirv         = Vulkan::GLSLANG::createFromString(source, vk::ShaderStageFlagBits::eFragment);
    ASSERT_FALSE(spirv.empty());
    EXPECT_EQ(spirv[0], spirv_magic);
}

TEST_F(VulkanShader, CompileErrorReturnsNoCode)
{
    std::string source = R"(#version 450
void main()
{
    this is not glsl;
}
)";
    auto spirv         = Vulkan::GLSLANG::createFromString(source, vk::ShaderStageFlagBits::eFragment);
    EXPECT_TRUE(spirv.empty());
}

TEST_F(VulkanShader, LoadFile)
{
    auto spirv = Vulkan::GLSLANG::loadGLSL("vulkan/coloredAsset.vert", vk::ShaderStageFlagBits::eVertex);
    ASSERT_FALSE(spirv.empty());
    EXPECT_EQ(spirv[0], spirv_magic);

    EXPECT_TRUE(Vulkan::GLSLANG::loadGLSL("vulkan/does_not_exist.vert", vk::ShaderStageFlagBits::eVertex).empty());
}

TEST_F(VulkanShader, Injection)
{
    std::string source = "#version 450\nvoid main() {}\n";
    ASSERT_TRUE(Vulkan::GLSLANG::addInjectionAfterVersion(source, "#define INJECTED 1"));
    auto version = source.find("#version");
    auto define  = source.find("#define INJECTED 1");
    auto main    = source.find("void main");
    ASSERT_NE(define, std::string::npos);
    EXPECT_LT(version, define);
    EXPECT_LT(define, main);

    std::string no_version = "void main() {}\n";
    EXPECT_FALSE(Vulkan::GLSLANG::addInjectionAfterVersion(no_version, "#define INJECTED 1"));
}

TEST_F(VulkanShader, ShaderPipeline)
{
    Vulkan::GraphicsShaderPipeline pipeline;
    EXPECT_FALSE(pipeline.valid());
    EXPECT_TRUE(pipeline.load(base().device, {"vulkan/coloredAsset.vert", "vulkan/coloredAsset.frag"}));
    EXPECT_TRUE(pipeline.valid());
    pipeline.destroy();
    EXPECT_FALSE(pipeline.valid());
}
