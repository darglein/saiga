/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/UnifiedMeshBuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shaderStorageBuffer.h"
#include "saiga/opengl/templatedBuffer.h"

#include "opengl_test_context.h"

#include <type_traits>

using namespace Saiga;

class OpenGLBuffer : public OpenGLTest
{
};

TEST_F(OpenGLBuffer, CreateAndGet)
{
    std::vector<int> data = {1, -2, 3, -4, 5, -6, 7};

    TemplatedBuffer<int> buffer(GL_ARRAY_BUFFER);
    buffer.create(data, GL_STATIC_DRAW);
    EXPECT_EQ(buffer.Size(), static_cast<int>(data.size()));
    EXPECT_NE(buffer.getBufferObject(), 0u);

    std::vector<int> result(data.size(), 0);
    buffer.get(result);
    EXPECT_EQ(result, data);
}

TEST_F(OpenGLBuffer, UpdateWithOffset)
{
    std::vector<float> data(10, 0.f);
    TemplatedBuffer<float> buffer(GL_ARRAY_BUFFER);
    buffer.create(data, GL_DYNAMIC_DRAW);

    std::vector<float> part = {1.5f, 2.5f, 3.5f};
    buffer.update(part, 4);

    std::vector<float> result(10, -1.f);
    buffer.get(result);
    std::vector<float> expected = {0, 0, 0, 0, 1.5f, 2.5f, 3.5f, 0, 0, 0};
    EXPECT_EQ(result, expected);

    // Read only a part
    std::vector<float> slice(2);
    buffer.get(slice, 5);
    EXPECT_EQ(slice, std::vector<float>({2.5f, 3.5f}));
}

TEST_F(OpenGLBuffer, Resize)
{
    std::vector<int> data = {1, 2, 3};
    TemplatedBuffer<int> buffer(GL_ARRAY_BUFFER);
    buffer.create(data, GL_DYNAMIC_DRAW);

    buffer.resize(100);
    EXPECT_EQ(buffer.Size(), 100);

    GLint64 size = 0;
    buffer.bind();
    glGetBufferParameteri64v(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    EXPECT_EQ(size, static_cast<GLint64>(100 * sizeof(int)));

    // Resizing to a smaller size keeps the buffer.
    buffer.resize(10);
    EXPECT_EQ(buffer.Size(), 100);
}

TEST_F(OpenGLBuffer, MapBuffer)
{
    std::vector<unsigned int> data(64);
    for (size_t i = 0; i < data.size(); ++i) data[i] = i * i;

    TemplatedShaderStorageBuffer<unsigned int> buffer;
    buffer.create(data, GL_DYNAMIC_DRAW);

    buffer.bind();
    auto* ptr = static_cast<unsigned int*>(buffer.mapBuffer(GL_READ_WRITE));
    ASSERT_NE(ptr, nullptr);
    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_EQ(ptr[i], data[i]);
        ptr[i] += 1;
    }
    buffer.unmapBuffer();

    std::vector<unsigned int> result(data.size());
    buffer.get(result);
    for (size_t i = 0; i < data.size(); ++i) EXPECT_EQ(result[i], data[i] + 1);
}

TEST_F(OpenGLBuffer, CopyConstructor)
{
    std::vector<int> data = {9, 8, 7, 6};
    TemplatedBuffer<int> buffer(GL_ARRAY_BUFFER);
    buffer.create(data, GL_STATIC_DRAW);

    Buffer copy(buffer);
    EXPECT_NE(copy.getBufferObject(), buffer.getBufferObject());

    std::vector<int> result(data.size());
    copy.getBuffer(result.data(), result.size() * sizeof(int), 0);
    EXPECT_EQ(result, data);
}

TEST_F(OpenGLBuffer, ShaderStorageLimits)
{
    EXPECT_GE(ShaderStorageBuffer::getMaxShaderStorageBufferBindings(), 8);
    EXPECT_GE(ShaderStorageBuffer::getMaxShaderStorageBlockSize(), 1 << 24);
}

// Copying a buffer without a GL object read from buffer 0 (GL_INVALID_OPERATION) and created a new GL buffer.
TEST_F(OpenGLBuffer, CopyEmptyBuffer)
{
    TemplatedBuffer<int> buffer(GL_ARRAY_BUFFER);
    Buffer copy(buffer);
    EXPECT_EQ(copy.getBufferObject(), 0u);
    EXPECT_EQ(copy.size, 0u);
}

// updateFromMesh passed the vertex count and offset as byte count and byte offset.
TEST_F(OpenGLBuffer, IndexedVertexBufferUpdateFromMesh)
{
    TriangleMesh<VertexNT, GLuint> mesh;
    for (int i = 0; i < 6; ++i) mesh.vertices.push_back(VertexNT(vec3(i, 0, 0)));
    mesh.addFace(0, 1, 2);
    mesh.addFace(3, 4, 5);

    IndexedVertexBuffer<VertexNT, GLuint> buffer;
    buffer.fromMesh(mesh, GL_DYNAMIC_DRAW);
    EXPECT_EQ(buffer.VertexCount(), 6);
    EXPECT_EQ(buffer.IndexCount(), 6);

    for (int i = 0; i < 6; ++i) mesh.vertices[i].position(0) = 10 + i;
    buffer.updateFromMesh(mesh, 3, 2);

    std::vector<VertexNT> result(6);
    buffer.VertexBuffer<VertexNT>::get(result);
    std::vector<float> expected = {0, 1, 12, 13, 14, 5};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(result[i].position(0), expected[i]) << i;
}

// The draw mode of a VertexBuffer was uninitialized until setDrawMode was called.
TEST_F(OpenGLBuffer, VertexBufferDefaultDrawMode)
{
    VertexBuffer<Vertex> buffer;
    EXPECT_EQ(buffer.getDrawMode(), GL_TRIANGLES);
}

// A copy of a VertexBuffer shared the VAO of the original and deleted it twice.
static_assert(!std::is_copy_constructible_v<VertexBuffer<Vertex>>, "VertexBuffer must not be copyable");
static_assert(!std::is_copy_assignable_v<VertexBuffer<Vertex>>, "VertexBuffer must not be copyable");
static_assert(!std::is_copy_constructible_v<IndexedVertexBuffer<Vertex, GLuint>>,
              "IndexedVertexBuffer must not be copyable");

// The constructor of UnifiedMeshBuffer indexed into the empty index vector of a mesh without faces or lines.
TEST_F(OpenGLBuffer, UnifiedMeshBufferEmptyMesh)
{
    UnifiedMesh mesh;
    for (GLenum mode : {GL_TRIANGLES, GL_LINES, GL_POINTS})
    {
        UnifiedMeshBuffer buffer(mesh, mode);
        EXPECT_EQ(buffer.num_elements, 0);
        buffer.BindAndDraw();
    }
}
