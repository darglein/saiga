/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/Shader/GLSL.h"
#include "saiga/vulkan/Shader/ShaderPipeline.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/Pipeline.h"

#include <filesystem>
#include <fstream>
#include <thread>

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

// Regression: an include that was missing while the shader was reloaded (e.g. while an editor saves it) was dropped
// from the watched files, so later changes of the include did not reload the shader anymore.
TEST_F(VulkanShader, ReloadAfterMissingInclude)
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_vulkan_shader_reload";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    auto write = [](const std::filesystem::path& file, const std::string& content) { std::ofstream(file) << content; };
    auto include = dir / "color.glsl";
    auto shader  = dir / "reload.frag";
    write(include, "vec4 color() { return vec4(1, 0, 0, 1); }\n");
    write(shader,
          "#version 450\n#extension GL_GOOGLE_include_directive : require\n#include \"color.glsl\"\n"
          "layout(location = 0) out vec4 outColor;\nvoid main() { outColor = color(); }\n");

    Vulkan::ShaderModule module;
    ASSERT_TRUE(module.loadGLSL(base().device, vk::ShaderStageFlagBits::eFragment, shader.string()));

    // autoReload() looks at the files at most every 0.5 s.
    auto wait = []() { std::this_thread::sleep_for(std::chrono::milliseconds(600)); };

    std::filesystem::remove(include);
    wait();
    EXPECT_FALSE(module.autoReload());
    EXPECT_TRUE(module.valid());

    write(include, "vec4 color() { return vec4(0, 1, 0, 1); }\n");
    wait();
    EXPECT_TRUE(module.autoReload());

    module.destroy();
    std::filesystem::remove_all(dir);
}

// S17 (docs/engine_design.md §10.2): the shaders are compiled for Vulkan 1.3 / SPIR-V 1.6
TEST_F(VulkanShader, SpirvVersion16)
{
    std::string source = R"(#version 450
layout(location = 0) out vec4 outColor;
void main()
{
    outColor = vec4(1);
}
)";
    auto spirv         = Vulkan::GLSLANG::createFromString(source, vk::ShaderStageFlagBits::eFragment);
    ASSERT_GE(spirv.size(), 2u);
    // The version word: 0x00MMmm00
    EXPECT_EQ(spirv[1], 0x00010600u);
    auto file = Vulkan::GLSLANG::loadGLSL("vulkan/coloredAsset.vert", vk::ShaderStageFlagBits::eVertex);
    ASSERT_GE(file.size(), 2u);
    EXPECT_EQ(file[1], 0x00010600u);
}

// Every stage file of shader/vulkan/ compiles (the .glsl includes are compiled through them; includes are resolved
// through the shader directory as the loader does)
TEST_F(VulkanShader, EveryStageFileCompiles)
{
    namespace fs        = std::filesystem;
    const fs::path root = fs::path(SAIGA_PROJECT_SOURCE_DIR) / "shader";
    int compiled        = 0;
    for (const auto& entry : fs::recursive_directory_iterator(root / "vulkan"))
    {
        if (!entry.is_regular_file()) continue;
        const std::string ext = entry.path().extension().string();
        vk::ShaderStageFlagBits stage;
        if (ext == ".vert")
            stage = vk::ShaderStageFlagBits::eVertex;
        else if (ext == ".frag")
            stage = vk::ShaderStageFlagBits::eFragment;
        else if (ext == ".comp")
            stage = vk::ShaderStageFlagBits::eCompute;
        else
            continue;
        const std::string relative = fs::relative(entry.path(), root).generic_string();
        auto spirv                 = Vulkan::GLSLANG::loadGLSL(relative, stage);
        EXPECT_GE(spirv.size(), 2u) << relative;
        if (spirv.size() >= 2)
        {
            EXPECT_EQ(spirv[1], 0x00010600u) << relative;
        }
        compiled++;
    }
    EXPECT_GT(compiled, 20);
}

// SPIR-V 1.6 turns discard into OpDemoteToHelperInvocation (shaderDemoteToHelperInvocation, enabled by the device):
// a pipeline with discard is validation clean (the layer aborts on an error) and discards
TEST_F(VulkanShader, DiscardPipeline)
{
    static const std::string vert = R"(#version 450
void main()
{
    vec2 uv     = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.5, 1.0);
}
)";
    static const std::string frag = R"(#version 450
layout(location = 0) out vec4 outColor;
void main()
{
    if (gl_FragCoord.x < 32.0) discard;
    outColor = vec4(1, 0, 0, 1);
}
)";
    auto spirv                    = Vulkan::GLSLANG::createFromString(frag, vk::ShaderStageFlagBits::eFragment);
    ASSERT_GE(spirv.size(), 2u);
    // OpDemoteToHelperInvocation (opcode 5380) instead of OpKill (252)
    bool demote = false, kill = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t op = spirv[i] & 0xffff, words = spirv[i] >> 16;
        demote |= op == 5380;
        kill |= op == 252;
        i += words > 0 ? words : 1;
    }
    EXPECT_TRUE(demote);
    EXPECT_FALSE(kill);

    Vulkan::Pipeline pipeline;
    pipeline.init(base(), 0);
    ASSERT_TRUE(pipeline.shaderPipeline.loadGLSLStrings(
        base().device, {{vk::ShaderStageFlagBits::eVertex, vert}, {vk::ShaderStageFlagBits::eFragment, frag}}));
    Vulkan::PipelineInfo info;
    info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
    info.depthStencilState.depthTestEnable  = VK_FALSE;
    info.depthStencilState.depthWriteEnable = VK_FALSE;
    pipeline.create(renderer->renderingFormats(), info);
    app.onRender = [&](vk::CommandBuffer cmd)
    {
        if (pipeline.bind(cmd)) cmd.draw(3, 1, 0, 0);
    };
    auto img = RenderFrame();
    ASSERT_EQ(img.w, width);
    EXPECT_EQ(img(10, 10), ClearColor());
    EXPECT_EQ(img(10, 50), ucvec4(255, 0, 0, 255));
    renderer->waitIdle();
    app = VulkanTestApp();
    pipeline.destroy();
}

// Regression (CodeRabbit on S17): with SPIR-V 1.6 glslang compiles local_size_x_id (a specialized workgroup size) to
// the execution mode LocalSizeId, which needs the maintenance4 feature; the device did not enable it (a validation
// error at pipeline creation, the layer aborts). The pipeline is created and dispatched with the default size 4.
TEST_F(VulkanShader, LocalSizeIdPipeline)
{
    static const std::string comp = R"(#version 450
layout(local_size_x = 4, local_size_x_id = 0) in;
layout(set = 0, binding = 0) buffer Numbers { uint numbers[]; };
void main()
{
    numbers[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x + 1;
}
)";
    auto spirv                    = Vulkan::GLSLANG::createFromString(comp, vk::ShaderStageFlagBits::eCompute);
    ASSERT_GE(spirv.size(), 2u);
    // OpExecutionModeId (opcode 331) with LocalSizeId (38)
    bool localSizeId = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t op = spirv[i] & 0xffff, words = spirv[i] >> 16;
        localSizeId |= op == 331 && words >= 3 && spirv[i + 2] == 38;
        i += words > 0 ? words : 1;
    }
    EXPECT_TRUE(localSizeId);

    std::vector<uint32_t> data(8, 0);
    const size_t size = data.size() * sizeof(uint32_t);
    Vulkan::Buffer storageBuffer;
    storageBuffer.createBuffer(base(), size, vk::BufferUsageFlagBits::eStorageBuffer,
                               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    storageBuffer.upload(data.data(), size);

    Vulkan::ComputePipeline pipeline;
    pipeline.init(base(), 1);
    pipeline.addDescriptorSetLayout(
        {{0, {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}}});
    ASSERT_TRUE(pipeline.shaderPipeline.loadGLSLStrings(base().device, {{vk::ShaderStageFlagBits::eCompute, comp}}));
    pipeline.create();
    auto set        = pipeline.createDescriptorSet();
    auto bufferInfo = storageBuffer.createInfo();
    base().device.updateDescriptorSets(
        {vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufferInfo, nullptr)},
        nullptr);

    bool bound = false;
    Submit(
        [&](vk::CommandBuffer cmd)
        {
            bound = pipeline.bind(cmd);
            if (!bound) return;
            pipeline.bindDescriptorSet(cmd, set);
            cmd.dispatch(2, 1, 1);
            vk::MemoryBarrier2 toHost(vk::PipelineStageFlagBits2::eComputeShader,
                                      vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eHost,
                                      vk::AccessFlagBits2::eHostRead);
            cmd.pipelineBarrier2(vk::DependencyInfo({}, toHost, nullptr, nullptr));
        });
    ASSERT_TRUE(bound);
    storageBuffer.download(data.data(), size);
    for (uint32_t i = 0; i < data.size(); ++i) EXPECT_EQ(data[i], i + 1) << i;
    pipeline.destroy();
    storageBuffer.destroy();
}
