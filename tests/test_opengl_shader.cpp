/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/shader/shader.h"
#include "saiga/opengl/shaderStorageBuffer.h"
#include "saiga/opengl/texture/ArrayTexture2D.h"
#include "saiga/opengl/texture/Texture2D.h"
#include "saiga/opengl/texture/Texture3D.h"
#include "saiga/opengl/uniformBuffer.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "opengl_test_context.h"

using namespace Saiga;

// A triangle covering the whole viewport, from gl_VertexID (no vertex buffer needed).
static const char* fullscreen_color_shader = R"(
##GL_VERTEX_SHADER
#version 330
void main()
{
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}

##GL_FRAGMENT_SHADER
#version 330
uniform vec4 color;
layout(location = 0) out vec4 out_color;
void main()
{
    out_color = color;
}
)";

// out[i] = in[i] * in[i] + offset
static const char* square_compute_shader = R"(
##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer In { uint data_in[]; };
layout(std430, binding = 1) writeonly buffer Out { uint data_out[]; };
uniform uint offset;
uniform uint n;
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    data_out[i] = data_in[i] * data_in[i] + offset;
}
)";

// Writes the pixel coordinate into an rgba8ui image.
static const char* image_compute_shader = R"(
##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 2, rgba8ui) uniform writeonly uimage2D dst;
void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    imageStore(dst, p, uvec4(p.x, p.y, p.x + p.y, 255));
}
)";

class OpenGLShader : public OpenGLTest
{
};

TEST_F(OpenGLShader, CompileError)
{
    Shader shader;
    shader.init_from_string("##GL_FRAGMENT_SHADER\n#version 330\nvoid main() { this is not glsl }\n");
    EXPECT_EQ(shader.program, 0u);
    EXPECT_FALSE(shader.bind());
}

// init returned true on a link error.
TEST_F(OpenGLShader, LinkErrorReturnsFalse)
{
    Shader shader;
    // Compiles, but the call of the undefined function f fails to link.
    EXPECT_FALSE(shader.init_from_string(
        "##GL_FRAGMENT_SHADER\n#version 330\nvec4 f();\nout vec4 c;\nvoid main() { c = f(); }\n"));
    EXPECT_EQ(shader.program, 0u);

    EXPECT_TRUE(shader.init_from_string(fullscreen_color_shader));
    EXPECT_NE(shader.program, 0u);
}

// A second init (as on a reload) leaked the GL program of the first one.
TEST_F(OpenGLShader, ReinitDeletesOldProgram)
{
    Shader shader;
    ASSERT_TRUE(shader.init_from_string(fullscreen_color_shader));
    GLuint first = shader.program;
    ASSERT_NE(first, 0u);

    ASSERT_TRUE(shader.init_from_string(fullscreen_color_shader));
    EXPECT_NE(shader.program, 0u);
    // The driver may reuse the name of the deleted program.
    EXPECT_TRUE(shader.program == first || glIsProgram(first) == GL_FALSE);
}

// A reinit of a bound shader left the deleted program bound and Shader::boundShader pointing to it.
TEST_F(OpenGLShader, ReinitWhileBound)
{
    Shader shader;
    ASSERT_TRUE(shader.init_from_string(fullscreen_color_shader));
    ASSERT_TRUE(shader.bind());

    ASSERT_TRUE(shader.init_from_string(fullscreen_color_shader));
    EXPECT_EQ(Shader::getBoundShader(), 0u);
    GLint current = -1;
    glGetIntegerv(GL_CURRENT_PROGRAM, &current);
    EXPECT_EQ(current, 0);

    ASSERT_TRUE(shader.bind());
    EXPECT_TRUE(shader.isBound());
    shader.unbind();
    EXPECT_EQ(Shader::getBoundShader(), 0u);
}

TEST_F(OpenGLShader, RenderFullscreenTriangle)
{
    int w = 16, h = 12;

    Shader shader;
    shader.init_from_string(fullscreen_color_shader);
    ASSERT_NE(shader.program, 0u);
    GLint color_location = shader.getUniformLocation("color");
    ASSERT_GE(color_location, 0);

    auto tex = std::make_shared<Texture>();
    tex->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
    Framebuffer fb;
    fb.create();
    fb.attachTexture(tex);
    fb.drawToAll();
    fb.check();

    glViewport(0, 0, w, h);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);

    // The core profile needs a vertex array object for a draw call.
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    ASSERT_TRUE(shader.bind());
    shader.upload(color_location, vec4(1.f, 0.5f, 0.f, 1.f));
    glDrawArrays(GL_TRIANGLES, 0, 3);
    shader.unbind();

    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);

    auto img = ReadPixels(w, h);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            auto c = img(y, x);
            EXPECT_EQ(c(0), 255);
            EXPECT_NEAR(c(1), 128, 1);
            EXPECT_EQ(c(2), 0);
            EXPECT_EQ(c(3), 255);
        }
    }

    fb.unbind();
}

TEST_F(OpenGLShader, ComputeShaderBuffer)
{
    int n = 1000;
    std::vector<unsigned int> data(n);
    for (int i = 0; i < n; ++i) data[i] = i;

    std::vector<unsigned int> zeros(n, 0);
    TemplatedShaderStorageBuffer<unsigned int> in_buffer, out_buffer;
    in_buffer.create(data, GL_STATIC_DRAW);
    out_buffer.create(zeros, GL_DYNAMIC_READ);

    Shader shader;
    shader.init_from_string(square_compute_shader);
    ASSERT_NE(shader.program, 0u);
    EXPECT_EQ(shader.getComputeWorkGroupSize(), uvec3(64, 1, 1));

    in_buffer.bind(0);
    out_buffer.bind(1);
    ASSERT_TRUE(shader.bind());
    shader.upload(shader.getUniformLocation("offset"), 7u);
    shader.upload(shader.getUniformLocation("n"), static_cast<unsigned int>(n));
    auto groups = shader.getNumGroupsCeil(uvec3(n, 1, 1));
    EXPECT_EQ(groups, uvec3(16, 1, 1));
    shader.dispatchCompute(groups);
    shader.unbind();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<unsigned int> result(n);
    out_buffer.get(result);
    for (int i = 0; i < n; ++i) EXPECT_EQ(result[i], static_cast<unsigned int>(i * i + 7)) << i;
}

TEST_F(OpenGLShader, ComputeShaderImage)
{
    int w = 20, h = 13;
    Texture2D tex;
    tex.create(w, h, GL_RGBA_INTEGER, GL_RGBA8UI, GL_UNSIGNED_BYTE);

    Shader shader;
    shader.init_from_string(image_compute_shader);
    ASSERT_NE(shader.program, 0u);

    tex.bindImageTexture(2, GL_WRITE_ONLY);
    ASSERT_TRUE(shader.bind());
    shader.dispatchCompute(shader.getNumGroupsCeil(uvec3(w, h, 1)));
    shader.unbind();
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

    TemplatedImage<ucvec4> img(h, w);
    ASSERT_TRUE(tex.download(img.data()));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) EXPECT_EQ(img(y, x), ucvec4(x, y, x + y, 255)) << x << " " << y;
}

// Copies the texels of an integer texture into a buffer with texelFetch.
static const char* integer_texture_compute_shader = R"(
##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(location = 0) uniform usampler2D src;
layout(std430, binding = 0) writeonly buffer Out { uint data_out[]; };
void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = textureSize(src, 0);
    if (p.x >= size.x || p.y >= size.y) return;
    data_out[p.y * size.x + p.x] = texelFetch(src, p, 0).r;
}
)";

// Integer textures were created with GL_LINEAR filtering, which makes them incomplete: every fetch returned 0.
TEST_F(OpenGLShader, SampleIntegerTexture)
{
    int w = 5, h = 3;
    TemplatedImage<unsigned int> img(h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) img(y, x) = 1000 * y + x + 1;

    Texture2D tex(img);

    std::vector<unsigned int> zeros(w * h, 0);
    TemplatedShaderStorageBuffer<unsigned int> out_buffer;
    out_buffer.create(zeros, GL_DYNAMIC_READ);

    Shader shader;
    ASSERT_TRUE(shader.init_from_string(integer_texture_compute_shader));
    out_buffer.bind(0);
    ASSERT_TRUE(shader.bind());
    shader.upload(0, tex, 0);
    shader.dispatchCompute(shader.getNumGroupsCeil(uvec3(w, h, 1)));
    shader.unbind();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<unsigned int> result(w * h);
    out_buffer.get(result);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) EXPECT_EQ(result[y * w + x], img(y, x)) << x << " " << y;
}

static const char* uniform_array_shader = R"(
##GL_VERTEX_SHADER
#version 330
uniform ivec2 iv[3];
uniform vec3 v3[2];
uniform mat3 m3[2];
uniform mat4 m4;
out vec4 v;
void main()
{
    v = vec4(iv[0].x + iv[1].y + iv[2].x, v3[0].x + v3[1].z, m3[0][1][2] + m3[1][2][0], 0) + m4[3];
    gl_Position = vec4(0);
}

##GL_FRAGMENT_SHADER
#version 330
in vec4 v;
out vec4 c;
void main()
{
    c = v;
}
)";

// UniformHelper<ivec2> ignored the count and only uploaded the first element of an array.
TEST_F(OpenGLShader, UploadUniformArrays)
{
    Shader shader;
    ASSERT_TRUE(shader.init_from_string(uniform_array_shader));
    ASSERT_TRUE(shader.bind());

    auto location = [&](const std::string& name) { return shader.getUniformLocation(name.c_str()); };

    ivec2 iv[3] = {ivec2(1, 2), ivec2(3, 4), ivec2(5, 6)};
    shader.upload(location("iv"), 3, iv);
    for (int i = 0; i < 3; ++i)
    {
        GLint value[2] = {0, 0};
        glGetUniformiv(shader.program, location("iv[" + std::to_string(i) + "]"), value);
        EXPECT_EQ(value[0], iv[i].x()) << i;
        EXPECT_EQ(value[1], iv[i].y()) << i;
    }

    vec3 v3[2] = {vec3(1, 2, 3), vec3(4, 5, 6)};
    shader.upload(location("v3"), 2, v3);
    for (int i = 0; i < 2; ++i)
    {
        float value[3] = {};
        glGetUniformfv(shader.program, location("v3[" + std::to_string(i) + "]"), value);
        for (int j = 0; j < 3; ++j) EXPECT_EQ(value[j], v3[i](j)) << i << " " << j;
    }

    // Matrices are column major on both sides
    mat3 m3[2];
    for (int i = 0; i < 2; ++i)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) m3[i](r, c) = 100 * i + 10 * r + c;
    shader.upload(location("m3"), 2, m3);
    for (int i = 0; i < 2; ++i)
    {
        float value[9] = {};
        glGetUniformfv(shader.program, location("m3[" + std::to_string(i) + "]"), value);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) EXPECT_EQ(value[c * 3 + r], m3[i](r, c)) << i << " " << r << " " << c;
    }

    mat4 m4;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) m4(r, c) = 10 * r + c;
    shader.upload(location("m4"), m4);
    float value[16] = {};
    glGetUniformfv(shader.program, location("m4"), value);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) EXPECT_EQ(value[c * 4 + r], m4(r, c)) << r << " " << c;

    shader.unbind();
}

// Samples the center of every voxel of a 3D texture and writes the red channel (0..255) into a buffer.
static const char* texture3d_compute_shader = R"(
##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(location = 0) uniform sampler3D src;
layout(std430, binding = 0) writeonly buffer Out { uint data_out[]; };
void main()
{
    ivec3 p = ivec3(gl_GlobalInvocationID.xyz);
    ivec3 size = textureSize(src, 0);
    if (any(greaterThanEqual(p, size))) return;
    vec4 c = texture(src, (vec3(p) + 0.5) / vec3(size));
    data_out[(p.z * size.y + p.y) * size.x + p.x] = uint(round(c.r * 255.0));
}
)";

// Texture3D::fromImage did not set the filter parameters. The default minification filter uses mipmaps, which
// fromImage does not create, so the texture was incomplete and every sample returned 0.
TEST_F(OpenGLShader, SampleTexture3DFromImage)
{
    int w = 3, h = 2, d = 2;
    std::vector<Image> images;
    for (int z = 0; z < d; ++z)
    {
        TemplatedImage<ucvec4> img(h, w);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) img(y, x) = ucvec4(1 + x + 10 * y + 100 * z, 0, 0, 255);
        images.push_back(img);
    }
    Texture3D tex;
    ASSERT_TRUE(tex.fromImage(images));

    std::vector<unsigned int> zeros(w * h * d, 0);
    TemplatedShaderStorageBuffer<unsigned int> out_buffer;
    out_buffer.create(zeros, GL_DYNAMIC_READ);

    Shader shader;
    ASSERT_TRUE(shader.init_from_string(texture3d_compute_shader));
    out_buffer.bind(0);
    ASSERT_TRUE(shader.bind());
    shader.upload(0, tex, 0);
    shader.dispatchCompute(shader.getNumGroupsCeil(uvec3(w, h, d)));
    shader.unbind();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<unsigned int> result(w * h * d);
    out_buffer.get(result);
    for (int z = 0; z < d; ++z)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                EXPECT_EQ(result[(z * h + y) * w + x], 1u + x + 10 * y + 100 * z) << x << " " << y << " " << z;
}

// Writes VALUE (a define injected by the test) to the buffer.
static const char* injection_compute_shader = R"(##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) writeonly buffer Out { uint data_out[]; };
void main()
{
    data_out[0] = VALUE;
}
)";

static unsigned int RunInjectionShader(const ShaderCodeInjections& injections)
{
    std::vector<unsigned int> zeros(1, 0);
    TemplatedShaderStorageBuffer<unsigned int> out_buffer;
    out_buffer.create(zeros, GL_DYNAMIC_READ);

    Shader shader;
    if (!shader.init_from_string(injection_compute_shader, injections)) return 0;
    out_buffer.bind(0);
    if (!shader.bind()) return 0;
    shader.dispatchCompute(1, 1, 1);
    shader.unbind();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<unsigned int> result(1);
    out_buffer.get(result);
    return result[0];
}

TEST_F(OpenGLShader, CodeInjection)
{
    // Line 1 is the line after #version. Injections of other shader types are ignored.
    EXPECT_EQ(RunInjectionShader({{GL_COMPUTE_SHADER, "#define VALUE 42u", 1},
                                  {GL_FRAGMENT_SHADER, "#error not a fragment shader", 1}}),
              42u);
    // Later injections at the same line are inserted before the earlier ones.
    EXPECT_EQ(RunInjectionShader({{GL_COMPUTE_SHADER, "#define VALUE (A + B)", 1},
                                  {GL_COMPUTE_SHADER, "#define A 3u", 1},
                                  {GL_COMPUTE_SHADER, "#define B 4u", 1}}),
              7u);
}

// An injection behind the last line of a shader inserted past the end of the code vector.
TEST_F(OpenGLShader, CodeInjectionAtEnd)
{
    // Appended at the end, behind main(): the macro is only defined after its use, so the compilation fails.
    EXPECT_EQ(RunInjectionShader({{GL_COMPUTE_SHADER, "#define VALUE 1u", 1000}}), 0u);
}

static unsigned int RunShader(Shader& shader)
{
    std::vector<unsigned int> zeros(1, 0);
    TemplatedShaderStorageBuffer<unsigned int> out_buffer;
    out_buffer.create(zeros, GL_DYNAMIC_READ);
    out_buffer.bind(0);
    if (!shader.bind()) return 0;
    shader.dispatchCompute(1, 1, 1);
    shader.unbind();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<unsigned int> result(1);
    out_buffer.get(result);
    return result[0];
}

static void WriteFile(const std::filesystem::path& file, const std::string& content)
{
    std::filesystem::create_directories(file.parent_path());
    std::ofstream strm(file);
    strm << content;
}

// Loads a shader file with nested includes (relative to the including file) and reloads it after an included file
// changed.
TEST_F(OpenGLShader, LoadFileWithIncludesAndReload)
{
    auto dir = std::filesystem::temp_directory_path() / "saiga_test_opengl_shader_include";
    std::filesystem::remove_all(dir);

    WriteFile(dir / "main.glsl",
              "##GL_COMPUTE_SHADER\r\n"
              "#version 430\r\n"
              "layout(local_size_x = 1) in;\r\n"
              "layout(std430, binding = 0) writeonly buffer Out { uint data_out[]; };\r\n"
              "#include \"inc/a.glsl\"\r\n"
              "void main() { data_out[0] = A + B; }\r\n");
    WriteFile(dir / "inc/a.glsl", "#include \"b.glsl\"\nconst uint A = 10u;\n");
    WriteFile(dir / "inc/b.glsl", "const uint B = 5u;\n");

    Shader shader;
    ASSERT_TRUE(shader.init(dir / "main.glsl"));
    EXPECT_EQ(shader.dependent_files_and_date.size(), 3u);
    EXPECT_EQ(RunShader(shader), 15u);

    // Nothing changed
    GLuint program = shader.program;
    EXPECT_TRUE(shader.reload());
    EXPECT_EQ(shader.program, program);

    WriteFile(dir / "inc/b.glsl", "const uint B = 7u;\n");
    auto time = std::filesystem::last_write_time(dir / "main.glsl");
    std::filesystem::last_write_time(dir / "inc/b.glsl", time + std::chrono::seconds(10));
    EXPECT_TRUE(shader.reload());
    EXPECT_EQ(RunShader(shader), 17u);

    std::filesystem::remove_all(dir);
}

static const char* uniform_block_compute_shader = R"(
##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 1) in;
layout(std140) uniform Params
{
    vec4 scale;
    uint offset;
};
layout(std430, binding = 0) writeonly buffer Out { uint data_out[]; };
void main()
{
    data_out[0] = uint(scale.x * scale.w) + offset;
}
)";

TEST_F(OpenGLShader, UniformBuffer)
{
    Shader shader;
    ASSERT_TRUE(shader.init_from_string(uniform_block_compute_shader));

    GLuint block = shader.getUniformBlockLocation("Params");
    ASSERT_NE(block, GL_INVALID_INDEX);
    EXPECT_EQ(shader.getUniformBlockSize(block), 32);
    auto indices = shader.getUniformBlockIndices(block);
    ASSERT_EQ(indices.size(), 2u);
    auto offsets = shader.getUniformBlockOffset(indices);
    std::sort(offsets.begin(), offsets.end());
    EXPECT_EQ(offsets, std::vector<GLint>({0, 16}));

    struct Params
    {
        vec4 scale;
        unsigned int offset;
        unsigned int padding[3];
    } params;
    params.scale  = vec4(3, 0, 0, 5);
    params.offset = 100;

    UniformBuffer ubo;
    ubo.createGLBuffer(&params, sizeof(params), GL_DYNAMIC_DRAW);
    ubo.bind(3);
    shader.setUniformBlockBinding(block, 3);
    EXPECT_EQ(RunShader(shader), 115u);

    params.offset = 200;
    ubo.updateBuffer(&params, sizeof(params), 0);
    EXPECT_EQ(RunShader(shader), 215u);

    EXPECT_GE(UniformBuffer::getMaxUniformBlockSize(), 16384);
    EXPECT_GE(UniformBuffer::getMaxUniformBufferBindings(), 36);
}

// Sums the texels of all layers of an array texture.
static const char* array_texture_compute_shader = R"(
##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 1) in;
layout(location = 0) uniform sampler2DArray src;
layout(std430, binding = 0) writeonly buffer Out { uint data_out[]; };
void main()
{
    ivec3 size = textureSize(src, 0);
    float sum = 0;
    for (int z = 0; z < size.z; ++z)
        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x) sum += texelFetch(src, ivec3(x, y, z), 0).r;
    data_out[0] = uint(sum);
}
)";

TEST_F(OpenGLShader, ArrayTexture)
{
    int w = 3, h = 2, d = 4;
    ArrayTexture2D tex;
    tex.create(w, h, d, GL_RED, GL_R32F, GL_FLOAT);
    std::vector<float> data(w * h * d);
    for (size_t i = 0; i < data.size(); ++i) data[i] = i;
    tex.uploadSubImage(0, 0, 0, w, h, d, data.data());

    std::vector<float> result(data.size(), -1.f);
    ASSERT_TRUE(tex.download(result.data()));
    EXPECT_EQ(result, data);

    Shader shader;
    ASSERT_TRUE(shader.init_from_string(array_texture_compute_shader));
    ASSERT_TRUE(shader.bind());
    shader.upload(0, tex, 1);
    shader.unbind();
    unsigned int n = data.size();
    EXPECT_EQ(RunShader(shader), n * (n - 1) / 2);
}
