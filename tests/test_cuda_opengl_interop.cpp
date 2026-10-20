/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/interop.h"
#include "saiga/opengl/templatedBuffer.h"

#include <cstring>

#include "opengl_test_context.h"
#include <type_traits>

using namespace Saiga;

// Regression test: Interop was copyable, so a copy unregistered the same resource a second time.
static_assert(!std::is_copy_constructible_v<CUDA::Interop>);
static_assert(!std::is_copy_assignable_v<CUDA::Interop>);
static_assert(std::is_nothrow_move_constructible_v<CUDA::Interop>);
static_assert(std::is_nothrow_move_assignable_v<CUDA::Interop>);

class CudaOpenGLInterop : public OpenGLTest
{
   protected:
    void SetUp() override
    {
        OpenGLTest::SetUp();
        if (IsSkipped()) return;
        auto vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        if (!vendor || !std::strstr(vendor, "NVIDIA"))
        {
            GTEST_SKIP() << "CUDA/OpenGL interop needs a context on an NVIDIA GPU";
        }
    }
};

static std::vector<int> ReadMapped(CUDA::Interop& interop, int n)
{
    std::vector<int> result(n, 0);
    EXPECT_EQ(interop.get_size(), n * sizeof(int));
    CHECK_CUDA_ERROR(cudaMemcpy(result.data(), interop.getDevicePtr(), n * sizeof(int), cudaMemcpyDeviceToHost));
    return result;
}

TEST_F(CudaOpenGLInterop, MoveBuffer)
{
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6};
    TemplatedBuffer<int> buffer(GL_ARRAY_BUFFER);
    buffer.create(data, GL_STATIC_DRAW);
    glFinish();

    CUDA::Interop a;
    a.registerBuffer(buffer.getBufferObject());

    // The moved-from object must not unregister the resource.
    CUDA::Interop b(std::move(a));
    b.map();
    EXPECT_EQ(ReadMapped(b, data.size()), data);

    // Move assignment of a mapped resource. The resource of c is unregistered first.
    std::vector<int> data2 = {7, 7, 7};
    TemplatedBuffer<int> buffer2(GL_ARRAY_BUFFER);
    buffer2.create(data2, GL_STATIC_DRAW);
    glFinish();
    CUDA::Interop c;
    c.registerBuffer(buffer2.getBufferObject());
    c = std::move(b);
    EXPECT_EQ(ReadMapped(c, data.size()), data);
    c.unmap();

    // The buffer can be registered again after c releases it.
    c.unregisterBuffer();
    CUDA::Interop d;
    d.registerBuffer(buffer.getBufferObject());
    d.map();
    EXPECT_EQ(ReadMapped(d, data.size()), data);
}
