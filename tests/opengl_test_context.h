/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"
#include "saiga/opengl/egl/offscreen_window.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/opengl_helper.h"

#include "gtest/gtest.h"

namespace Saiga
{
// Base fixture of the OpenGL tests: a headless EGL context (no window, no X server) for each test suite.
// Without an EGL display (no GPU and no Mesa) every test is skipped.
//
// Locally the context comes from the first EGL device (e.g. the NVIDIA GPU). To test on Mesa's software
// renderer, as CI does:
//   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json ctest -R test_opengl
class OpenGLTest : public ::testing::Test
{
   public:
    static void SetUpTestSuite() { context_available = CreateEGLContext(); }

    static void TearDownTestSuite()
    {
        if (context_available) TerminateEGLContext();
        context_available = false;
    }

   protected:
    void SetUp() override
    {
        if (!context_available) GTEST_SKIP() << "No EGL display available";
        // Clear the errors of an earlier test
        while (glGetError() != GL_NO_ERROR)
        {
        }
    }

    void TearDown() override
    {
        if (context_available)
        {
            EXPECT_EQ(glGetError(), GL_NO_ERROR);
        }
    }

    // Reads the color attachment 0 of the bound read framebuffer. Row 0 is the bottom row (OpenGL convention).
    static TemplatedImage<ucvec4> ReadPixels(int w, int h)
    {
        TemplatedImage<ucvec4> img(h, w);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img.data());
        return img;
    }

    static inline bool context_available = false;
};

}  // namespace Saiga
