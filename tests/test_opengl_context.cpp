/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/opengl_helper.h"

#include <filesystem>
#include <fstream>

#include "opengl_test_context.h"

using namespace Saiga;

class OpenGLContext : public OpenGLTest
{
};

TEST_F(OpenGLContext, IsInitialized)
{
    EXPECT_TRUE(OpenGLisInitialized());
    EXPECT_NE(glGetString(GL_VERSION), nullptr);
    EXPECT_NE(glGetString(GL_RENDERER), nullptr);
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
}

TEST_F(OpenGLContext, Version)
{
    // The tests use compute shaders and SSBOs
    int major = getVersionMajor();
    int minor = getVersionMinor();
    EXPECT_GE(major * 10 + minor, 43);
}

TEST_F(OpenGLContext, CoreProfile)
{
    GLint mask = 0;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
    EXPECT_TRUE(mask & static_cast<GLint>(GL_CONTEXT_CORE_PROFILE_BIT));
}

TEST_F(OpenGLContext, Extensions)
{
    auto extensions = getExtensions();
    EXPECT_EQ(static_cast<int>(extensions.size()), getExtensionCount());
    ASSERT_FALSE(extensions.empty());
    EXPECT_TRUE(hasExtension(extensions.front()));
    EXPECT_FALSE(hasExtension("GL_SAIGA_not_an_extension"));
}

// TerminateEGLContext() did not reset the "OpenGL initialized" flag, so a second CreateEGLContext() in the same
// process asserted in initOpenGL().
TEST_F(OpenGLContext, Recreate)
{
    TerminateEGLContext();
    EXPECT_FALSE(OpenGLisInitialized());
    context_available = CreateEGLContext();
    ASSERT_TRUE(context_available);
    EXPECT_TRUE(OpenGLisInitialized());

    GLuint tex = 0;
    glGenTextures(1, &tex);
    EXPECT_NE(tex, 0u);
    glDeleteTextures(1, &tex);
}

// The config file documents the profile values 'ANY' 'CORE' 'COMPATIBILITY', but the core profile was written and
// parsed as "saiga_core": a file with "CORE" selected the compatibility profile.
TEST(OpenGLParametersConfig, Profile)
{
    auto file = std::filesystem::temp_directory_path() / "saiga_test_opengl_parameters.ini";
    std::filesystem::remove(file);

    // Writes the defaults, then reads them back
    OpenGLParameters params;
    params.fromConfigFile(file.string());
    EXPECT_EQ(params.profile, OpenGLParameters::Profile::CORE);
    OpenGLParameters params2;
    params2.profile = OpenGLParameters::Profile::ANY;
    params2.fromConfigFile(file.string());
    EXPECT_EQ(params2.profile, OpenGLParameters::Profile::CORE);

    for (auto [name, profile] :
         {std::pair{"ANY", OpenGLParameters::Profile::ANY}, std::pair{"CORE", OpenGLParameters::Profile::CORE},
          std::pair{"COMPATIBILITY", OpenGLParameters::Profile::COMPATIBILITY}})
    {
        {
            std::ofstream strm(file);
            strm << "[opengl]\nprofile = " << name << "\n";
        }
        OpenGLParameters p;
        p.fromConfigFile(file.string());
        EXPECT_EQ(p.profile, profile) << name;
    }
    std::filesystem::remove(file);
}

// getOpenGLVendor searched for "ati" before "intel" and "mesa", so every vendor containing these letters (for example
// "Microsoft Corporation") was reported as Ati.
TEST(OpenGLVendorString, Classify)
{
    EXPECT_EQ(classifyOpenGLVendor("NVIDIA Corporation"), OpenGLVendor::Nvidia);
    EXPECT_EQ(classifyOpenGLVendor("ATI Technologies Inc."), OpenGLVendor::Ati);
    EXPECT_EQ(classifyOpenGLVendor("AMD"), OpenGLVendor::Ati);
    EXPECT_EQ(classifyOpenGLVendor("Advanced Micro Devices, Inc."), OpenGLVendor::Ati);
    EXPECT_EQ(classifyOpenGLVendor("Intel"), OpenGLVendor::Intel);
    EXPECT_EQ(classifyOpenGLVendor("Intel Corporation"), OpenGLVendor::Intel);
    EXPECT_EQ(classifyOpenGLVendor("Intel Open Source Technology Center"), OpenGLVendor::Intel);
    EXPECT_EQ(classifyOpenGLVendor("Mesa"), OpenGLVendor::Mesa);
    EXPECT_EQ(classifyOpenGLVendor("Mesa/X.org"), OpenGLVendor::Mesa);
    EXPECT_EQ(classifyOpenGLVendor("Microsoft Corporation"), OpenGLVendor::Unknown);
    EXPECT_EQ(classifyOpenGLVendor("Apple Computer, Inc."), OpenGLVendor::Unknown);
    EXPECT_EQ(classifyOpenGLVendor("Collabora Ltd"), OpenGLVendor::Unknown);
    EXPECT_EQ(classifyOpenGLVendor("VMware, Inc."), OpenGLVendor::Unknown);
    EXPECT_EQ(classifyOpenGLVendor(""), OpenGLVendor::Unknown);
}

TEST_F(OpenGLContext, Vendor)
{
    std::string vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    EXPECT_EQ(getOpenGLVendor(), classifyOpenGLVendor(vendor));
    std::cout << "Vendor: " << vendor << std::endl;
}
