/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/terrainmesh.h"

namespace Saiga
{
TerrainMesh::TerrainMesh() {}

std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMesh()
{
    unsigned int w = 100;
    unsigned int h = 100;


    return createGridMesh(w, h, vec2(2.0f / (w - 1), 2.0f / (h - 1)), make_vec2(1.0));
}


std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMesh2()
{
    return createGridMesh(m, m, make_vec2(1.0f / (m - 1)), make_vec2(0.5));
}



std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshFixUpV()
{
    return createGridMesh(3, m, make_vec2(1.0f / (m - 1)), make_vec2(0.5));
}



std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshFixUpH()
{
    return createGridMesh(m, 3, make_vec2(1.0f / (m - 1)), make_vec2(0.5));
}

std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshTrimSW()
{
    mesh_t* mesh = new mesh_t();

    unsigned int w = 2 * m + 1;
    unsigned int h = 2;
    vec2 d         = make_vec2(1.0f / (m - 1));
    vec2 o         = make_vec2(0.5);

    float dw = d[0];
    float dh = d[1];
    for (unsigned int y = 0; y < h; y++)
    {
        for (unsigned int x = 0; x < w; x++)
        {
            float fx = (float)x * dw - o[0];
            float fy = (float)y * dh - o[1];
            Vertex v(vec3(fx, 0.0f, fy));
            mesh->addVertex(v);
        }
    }


    for (unsigned int y = 0; y < h - 1; y++)
    {
        for (unsigned int x = 0; x < w - 1; x++)
        {
            GLuint quad[] = {y * w + x, (y + 1) * w + x, (y + 1) * w + x + 1, y * w + x + 1};
            mesh->addQuad(quad);
        }
    }

    int offset = mesh->vertices.size();

    w = 2;
    h = 2 * m;

    dw = d[0];
    dh = d[1];
    for (unsigned int y = 0; y < h; y++)
    {
        for (unsigned int x = 0; x < w; x++)
        {
            float fx = (float)x * dw - o[0];
            float fy = (float)(y + 1) * dh - o[1];
            Vertex v(vec3(fx, 0.0f, fy));
            mesh->addVertex(v);
        }
    }


    for (unsigned int y = 0; y < h - 1; y++)
    {
        for (unsigned int x = 0; x < w - 1; x++)
        {
            GLuint quad[] = {offset + y * w + x, offset + (y + 1) * w + x, offset + (y + 1) * w + x + 1,
                             offset + y * w + x + 1};
            mesh->addQuad(quad);
        }
    }

    return std::shared_ptr<TerrainMesh::mesh_t>(mesh);
}

std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshTrimSE()
{
    auto mesh = createMeshTrimSW();
    mesh->transform(rotate(radians(90.0f), vec3(0, 1, 0)));
    return mesh;
}

std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshTrimNW()
{
    auto mesh = createMeshTrimSW();
    mesh->transform(rotate(radians(-90.0f), vec3(0, 1, 0)));
    return mesh;
}


std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshTrimNE()
{
    auto mesh = createMeshTrimSW();
    mesh->transform(rotate(radians(180.0f), vec3(0, 1, 0)));
    return mesh;
}



std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshDegenerated()
{
    mesh_t* mesh = new mesh_t();
#if 0

    int w    = (n + 1) / 2;
    float dx = 2.0f / (m - 1);

    vec2 d[] = {vec2(dx, 0), vec2(dx, 0), vec2(0, dx), vec2(0, dx)};
    vec2 o[] = {vec2(0.5), vec2(0.5, -3.5 - (dx)), vec2(0.5), vec2(-3.5 - (dx), 0.5)};

    //    int orientation[] = {1,0,0,1};



    for (int i = 0; i < 4; i++)
    {
        int offset = mesh->vertices.size();

        for (int x = 0; x < w; x++)
        {
            float fx = (float)x * d[i][0] - o[i][0];
            float fy = (float)x * d[i][1] - o[i][1];
            Vertex v(vec3(fx, 0.0f, fy));
            mesh->addVertex(v);
            if (x < w - 1)
            {
                // add vertex between
                fx = fx + 0.5f * d[i][0];
                fy = fy + 0.5f * d[i][1];
                Vertex v(vec3(fx, 0.0f, fy));
                mesh->addVertex(v);
            }
        }


        for (int x = 0; x < w - 1; x++)
        {
            // add degenerated triangle
            unsigned int idx = 2 * x;
            GLuint face1[]   = {offset + idx, offset + idx + 1, offset + idx + 2};
            GLuint face2[]   = {offset + idx, offset + idx + 2, offset + idx + 1};

            //            if(orientation[i])
            mesh->addFace(face1);
            //            else
            mesh->addFace(face2);
        }
    }



#endif
    return std::shared_ptr<TerrainMesh::mesh_t>(mesh);
}



std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshCenter()
{
    int m = this->m * 2;
    return createGridMesh(m, m, make_vec2(1.0 / (m - 2)), make_vec2(0.5));
}


std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createGridMesh(unsigned int w, unsigned int h, vec2 d, vec2 o)
{
    mesh_t* mesh = new mesh_t();

    // creating uniform grid with w*h vertices
    // the resulting mesh will fill the quad (-1,0,-1) - (1,0,1)
    float dw = d[0];
    float dh = d[1];
    for (unsigned int y = 0; y < h; y++)
    {
        for (unsigned int x = 0; x < w; x++)
        {
            float fx = (float)x * dw - o[0];
            float fy = (float)y * dh - o[1];
            Vertex v(vec3(fx, 0.0f, fy));
            mesh->addVertex(v);
        }
    }


    for (unsigned int y = 0; y < h - 1; y++)
    {
        for (unsigned int x = 0; x < w - 1; x++)
        {
            GLuint quad[] = {y * w + x, (y + 1) * w + x, (y + 1) * w + x + 1, y * w + x + 1};
            mesh->addQuad(quad);
        }
    }

    return std::shared_ptr<TerrainMesh::mesh_t>(mesh);
}

}  // namespace Saiga
