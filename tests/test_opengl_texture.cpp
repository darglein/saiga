/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/texture/Texture2D.h"

#include "opengl_test_context.h"

using namespace Saiga;

class OpenGLTexture : public OpenGLTest
{
};

TEST_F(OpenGLTexture, RGBA8RoundTrip)
{
    int w = 7, h = 5;
    std::vector<unsigned char> data(w * h * 4);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<unsigned char>(i * 7 + 3);

    Texture2D tex;
    tex.create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, data.data());
    EXPECT_NE(tex.getId(), 0u);
    EXPECT_EQ(tex.getWidth(), w);
    EXPECT_EQ(tex.getHeight(), h);

    std::vector<unsigned char> result(data.size(), 0);
    ASSERT_TRUE(tex.download(result.data()));
    EXPECT_EQ(result, data);
}

TEST_F(OpenGLTexture, R32FRoundTrip)
{
    int w = 16, h = 9;
    std::vector<float> data(w * h);
    for (size_t i = 0; i < data.size(); ++i) data[i] = i * 0.25f - 10.f;

    Texture2D tex;
    tex.create(w, h, GL_RED, GL_R32F, GL_FLOAT, data.data());

    std::vector<float> result(data.size(), 0);
    ASSERT_TRUE(tex.download(result.data()));
    EXPECT_EQ(result, data);
}

TEST_F(OpenGLTexture, UploadSubImage)
{
    int w = 8, h = 8;
    std::vector<unsigned char> data(w * h, 0);

    Texture2D tex;
    tex.create(w, h, GL_RED, GL_R8, GL_UNSIGNED_BYTE, data.data());

    // Fill the 3x2 block at (2,4) with 200.
    std::vector<unsigned char> block(3 * 2, 200);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    tex.uploadSubImage(2, 4, 3, 2, block.data());

    std::vector<unsigned char> result(w * h, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    ASSERT_TRUE(tex.download(result.data()));
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            bool inside = x >= 2 && x < 5 && y >= 4 && y < 6;
            EXPECT_EQ(result[y * w + x], inside ? 200 : 0) << x << " " << y;
        }
    }
}

TEST_F(OpenGLTexture, Resize)
{
    Texture2D tex;
    tex.create(4, 4, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);

    std::vector<unsigned char> data(10 * 3 * 4, 42);
    tex.resize(10, 3, data.data());
    EXPECT_EQ(tex.getWidth(), 10);
    EXPECT_EQ(tex.getHeight(), 3);

    GLint w = 0, h = 0;
    tex.bind();
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
    tex.unbind();
    EXPECT_EQ(w, 10);
    EXPECT_EQ(h, 3);

    std::vector<unsigned char> result(data.size(), 0);
    ASSERT_TRUE(tex.download(result.data()));
    EXPECT_EQ(result, data);
}

TEST_F(OpenGLTexture, FromImage)
{
    TemplatedImage<ucvec4> img(3, 5);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) img(y, x) = ucvec4(x * 10, y * 20, 7, 255);

    Texture2D tex(img);
    EXPECT_EQ(tex.getWidth(), 5);
    EXPECT_EQ(tex.getHeight(), 3);

    TemplatedImage<ucvec4> result(3, 5);
    ASSERT_TRUE(tex.download(result.data()));
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) EXPECT_EQ(result(y, x), img(y, x)) << x << " " << y;
}

TEST_F(OpenGLTexture, FromImageFlipY)
{
    TemplatedImage<ucvec4> img(4, 2);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) img(y, x) = ucvec4(x, y, 0, 255);

    Texture2D tex(img, false, true);

    TemplatedImage<ucvec4> result(4, 2);
    ASSERT_TRUE(tex.download(result.data()));
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) EXPECT_EQ(result(y, x), img(img.h - 1 - y, x)) << x << " " << y;
}

// An RGB8 image with a width that is not a multiple of 4: its rows are padded, which must match OpenGL's
// default (un)pack alignment of 4.
TEST_F(OpenGLTexture, FromPaddedRGBImage)
{
    TemplatedImage<ucvec3> img(3, 5);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) img(y, x) = ucvec3(x + 1, y + 1, x * y);
    ASSERT_GT(img.pitchBytes, 3 * 5);

    Texture2D tex(img);

    TemplatedImage<ucvec3> result(3, 5);
    ASSERT_TRUE(tex.download(result.data()));
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) EXPECT_EQ(result(y, x), img(y, x)) << x << " " << y;
}
