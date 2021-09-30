/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UnifiedMeshBuffer.h"

namespace Saiga
{
UnifiedMeshBuffer::UnifiedMeshBuffer(UnifiedMesh mesh, GLenum draw_mode) : draw_mode(draw_mode)
{
    glGenVertexArrays(1, &gl_vao);
    glBindVertexArray(gl_vao);

    if (draw_mode == GL_TRIANGLES)
    {
        indices_per_element = 3;
        num_elements        = mesh.NumFaces();
        std::vector<uint32_t> indices_data(num_elements * indices_per_element);
        std::memcpy(&indices_data[0], &mesh.triangles[0], num_elements * sizeof(uint32_t) * indices_per_element);
        indices.create(indices_data, GL_STATIC_DRAW);
        indices.bind();
    }
    else if (draw_mode == GL_LINES)
    {
        indices_per_element = 2;
        num_elements        = mesh.lines.size();
        std::vector<uint32_t> indices_data(num_elements * indices_per_element);
        std::memcpy(&indices_data[0], &mesh.lines[0], num_elements * sizeof(uint32_t) * indices_per_element);
        indices.create(indices_data, GL_STATIC_DRAW);
        indices.bind();
    }
    else if (draw_mode == GL_POINTS)
    {
        is_indexed          = false;
        indices_per_element = 0;
        num_elements        = mesh.NumVertices();
    }
    else
    {
        SAIGA_EXIT_ERROR("invalid draw mode");
    }

    if (mesh.HasPosition())
    {
        glEnableVertexAttribArray(0);
        position.create(mesh.position, GL_STATIC_DRAW);
        position.bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), NULL);
    }

    if (mesh.HasNormal())
    {
        glEnableVertexAttribArray(1);
        normal.create(mesh.normal, GL_STATIC_DRAW);
        normal.bind();
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), NULL);
    }

    if (mesh.HasColor())
    {
        glEnableVertexAttribArray(2);
        color.create(mesh.color, GL_STATIC_DRAW);
        color.bind();
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(vec4), NULL);
    }

    if (mesh.HasTC())
    {
        glEnableVertexAttribArray(3);
        tc.create(mesh.texture_coordinates, GL_STATIC_DRAW);
        tc.bind();
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
    }

    if (mesh.HasBones())
    {
        glEnableVertexAttribArray(4);
        glEnableVertexAttribArray(5);
        bone_info.create(mesh.bone_info, GL_STATIC_DRAW);
        bone_info.bind();
        glVertexAttribIPointer(4, 4, GL_INT, sizeof(BoneInfo), (void*)(0 * sizeof(GLfloat)));
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(BoneInfo), (void*)(4 * sizeof(GLfloat)));
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //    std::cout << "UnifiedMeshBuffer " << indices.Size() << " " << position.Size() << " "
    //              << normal.Size() << " " << color.Size() << std::endl;
}


UnifiedMeshBuffer::~UnifiedMeshBuffer()
{
    glDeleteVertexArrays(1, &gl_vao);
    gl_vao = 0;
}


void UnifiedMeshBuffer::Draw(int offset, int count)
{
    //    if(count > num_triangles)  count = num_triangles;
    SAIGA_ASSERT(num_elements > 0);
    count = std::min(count, num_elements - offset);
    SAIGA_ASSERT(offset < num_elements);
    SAIGA_ASSERT(offset + count <= num_elements);
    SAIGA_ASSERT(offset >= 0 && offset + count <= num_elements);

    if (is_indexed)
    {
        glDrawElements(draw_mode, count * indices_per_element, GL_UNSIGNED_INT,
                       (void*)(intptr_t)(offset * indices_per_element * sizeof(uint32_t)));
    }
    else
    {
        glDrawArrays(draw_mode, offset, count);
    }
}
void UnifiedMeshBuffer::Bind()
{
    glBindVertexArray(gl_vao);
}
void UnifiedMeshBuffer::Unbind()
{
    glBindVertexArray(0);
}
}  // namespace Saiga
