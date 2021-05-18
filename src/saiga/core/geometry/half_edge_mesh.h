/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <iostream>
#include <map>

#include "triangle_mesh.h"
#include <unordered_map>

namespace Saiga
{
/*
 * Data structur for simple triangle meshes.
 * Can be turned into a IndexedVertexBuffer for drawing with OpenGL
 */

template <typename vertex_t, typename index_t>
class HalfEdgeMesh
{
   public:
    //    std::vector<vertex_t> vertices;
    //    std::vector<Face> faces;
    //    AABB boundingBox;


    struct HalfEdge
    {
        bool valid           = true;
        int oppositeHalfEdge = -1;
        int nextHalfEdge;
        int prevHalfEdge;
        // The vertex this half edge points to (not the start!!!)
        int vertex;
        int face;

        bool boundary() { return oppositeHalfEdge == -1; }
    };

    struct HalfVertex
    {
        bool valid = true;
        vertex_t v;
        // one outgoing halfedge
        int halfEdge;
    };

    struct HalfFace
    {
        bool valid = true;
        // one of the "inner" half edges
        int halfEdge;
    };

    std::vector<HalfEdge> edgeList;

    //    std::map< std::pair<int,int>, int > vertexEdges2;
    std::unordered_map<uint64_t, int> vertexEdges;
    //    std::map< uint64_t, int > vertexEdges;

    std::vector<HalfVertex> vertices;
    std::vector<HalfFace> faces;

    HalfEdgeMesh() {}
    HalfEdgeMesh(TriangleMesh<vertex_t, index_t>& ifs);

    void fromIFS(TriangleMesh<vertex_t, index_t>& ifs);
    void toIFS(TriangleMesh<vertex_t, index_t>& ifs);

    void clear();

    bool isValid();

    void halfEdgeCollapse(int he);
    void flipEdge(int he);
    void removeFace(int f);

    void getNeighbours(int vertex, std::vector<int>& neighs);
};

template <typename vertex_t, typename index_t>
HalfEdgeMesh<vertex_t, index_t>::HalfEdgeMesh(TriangleMesh<vertex_t, index_t>& ifs)
{
    fromIFS(ifs);
}

template <typename vertex_t, typename index_t>
void HalfEdgeMesh<vertex_t, index_t>::fromIFS(TriangleMesh<vertex_t, index_t>& ifs)
{
    clear();

    edgeList.resize(ifs.faces.size() * 3);
    vertices.resize(ifs.vertices.size());
    faces.resize(ifs.faces.size());

    vertexEdges.reserve(ifs.faces.size() * 3);

    for (int i = 0; i < (int)ifs.vertices.size(); ++i)
    {
        vertices[i].v = ifs.vertices[i];
    }



    for (int i = 0; i < (int)ifs.faces.size(); ++i)
    {
        typename TriangleMesh<vertex_t, index_t>::Face f = ifs.faces[i];

        // create 3 half edges for every face
        int heidx    = i * 3;
        HalfEdge& e1 = edgeList[heidx];
        HalfEdge& e2 = edgeList[heidx + 1];
        HalfEdge& e3 = edgeList[heidx + 2];

        // They all belong to the same face
        e1.face = i;
        e2.face = i;
        e3.face = i;

        e1.vertex = f(1);
        e2.vertex = f(2);
        e3.vertex = f(0);

        // make the circle pointers
        e1.nextHalfEdge = heidx + 1;
        e2.nextHalfEdge = heidx + 2;
        e3.nextHalfEdge = heidx + 0;

        e1.prevHalfEdge = heidx + 2;
        e2.prevHalfEdge = heidx + 0;
        e3.prevHalfEdge = heidx + 1;



        HalfFace& hf = faces[i];
        hf.halfEdge  = heidx;

        HalfVertex& hv1 = vertices[f(0)];
        HalfVertex& hv2 = vertices[f(1)];
        HalfVertex& hv3 = vertices[f(2)];

        hv1.halfEdge = heidx;
        hv2.halfEdge = heidx + 1;
        hv3.halfEdge = heidx + 2;

#if 1
        uint64_t mapidx;

        mapidx = std::min(f(0), f(1)) * vertices.size() + std::max(f(0), f(1));
        auto e = vertexEdges.find(mapidx);
        if (e == vertexEdges.end())
        {
            vertexEdges[mapidx] = heidx;
        }
        else
        {
            e1.oppositeHalfEdge                  = e->second;
            edgeList[e->second].oppositeHalfEdge = heidx;
        }


        mapidx = std::min(f(1), f(2)) * vertices.size() + std::max(f(1), f(2));
        e      = vertexEdges.find(mapidx);
        if (e == vertexEdges.end())
        {
            vertexEdges[mapidx] = heidx + 1;
        }
        else
        {
            e2.oppositeHalfEdge                  = e->second;
            edgeList[e->second].oppositeHalfEdge = heidx + 1;
        }


        mapidx = std::min(f(2), f(0)) * vertices.size() + std::max(f(2), f(0));
        e      = vertexEdges.find(mapidx);
        if (e == vertexEdges.end())
        {
            vertexEdges[mapidx] = heidx + 2;
        }
        else
        {
            e3.oppositeHalfEdge                  = e->second;
            edgeList[e->second].oppositeHalfEdge = heidx + 2;
        }

#else
        vertexEdges[std::pair<int, int>(f.v1, f.v2)] = heidx;
        vertexEdges[std::pair<int, int>(f.v2, f.v3)] = heidx + 1;
        vertexEdges[std::pair<int, int>(f.v3, f.v1)] = heidx + 2;
#endif
    }

#if 0

    //create opposite links
    for(int i = 0; i < (int)ifs.faces.size(); ++i)
    {
        typename TriangleMesh<vertex_t, index_t>::Face f = ifs.faces[i];

        int heidx = i * 3;


        auto e = vertexEdges.find(std::pair<int,int>(f.v2,f.v1));
        if( e != vertexEdges.end())
        {
            edgeList[heidx].oppositeHalfEdge = e->second;
        }

        e = vertexEdges.find(std::pair<int,int>(f.v3,f.v2));
        if( e != vertexEdges.end())
        {
            edgeList[heidx + 1].oppositeHalfEdge = e->second;
        }

        e = vertexEdges.find(std::pair<int,int>(f.v1,f.v3));
        if( e != vertexEdges.end())
        {
            edgeList[heidx + 2].oppositeHalfEdge = e->second;
        }
    }
#endif
}

template <typename vertex_t, typename index_t>
void HalfEdgeMesh<vertex_t, index_t>::toIFS(TriangleMesh<vertex_t, index_t>& ifs)
{
    ifs.faces.clear();


    ifs.vertices.resize(vertices.size());
    //    ifs.faces.resize(faces.size());
    for (int i = 0; i < (int)vertices.size(); ++i)
    {
        if (vertices[i].valid) ifs.vertices[i] = vertices[i].v;
    }

    for (int i = 0; i < (int)faces.size(); ++i)
    {
        HalfFace hf = faces[i];
        if (!hf.valid) continue;
        HalfEdge e1 = edgeList[faces[i].halfEdge];
        HalfEdge e2 = edgeList[e1.nextHalfEdge];
        HalfEdge e3 = edgeList[e2.nextHalfEdge];

        // make sure it's a triangle
        SAIGA_ASSERT(e3.nextHalfEdge == faces[i].halfEdge);

        typename TriangleMesh<vertex_t, index_t>::Face f;
        f(0) = e1.vertex;
        f(1) = e2.vertex;
        f(2) = e3.vertex;

        ifs.faces.push_back(f);
        //        ifs.faces[i] = f;
    }
}
template <typename vertex_t, typename index_t>
void HalfEdgeMesh<vertex_t, index_t>::clear()
{
    edgeList.clear();

    //    std::map< std::pair<int,int>, int > vertexEdges2;
    vertexEdges.clear();

    vertices.clear();
    faces.clear();
}

template <typename vertex_t, typename index_t>
bool HalfEdgeMesh<vertex_t, index_t>::isValid()
{
    // check opposites
    for (int i = 0; i < (int)edgeList.size(); ++i)
    {
        HalfEdge e = edgeList[i];
        if (!e.valid) continue;

        // edge of mesh
        if (e.oppositeHalfEdge == -1)
        {
            continue;
        }

        HalfEdge op = edgeList[e.oppositeHalfEdge];

        if (op.oppositeHalfEdge != i)
        {
            std::cout << "Opposite Half Edge Broken! " << i << "," << e.oppositeHalfEdge << " - " << op.oppositeHalfEdge
                      << std::endl;
            return false;
        }
    }

    // check circles
    for (int i = 0; i < (int)edgeList.size(); ++i)
    {
        HalfEdge e = edgeList[i];
        if (!e.valid) continue;

        if (edgeList[e.nextHalfEdge].prevHalfEdge != i)
        {
            std::cout << "prev broken" << std::endl;
            return false;
        }

        int f = e.face;
        if (f == -1 || !faces[f].valid)
        {
            std::cout << "valid half edge with broken face" << std::endl;
            return false;
        }


        int count = 0;
        while (e.nextHalfEdge != i)
        {
            e = edgeList[e.nextHalfEdge];
            if (count++ > 1000)
            {
                std::cout << "Half edge circle broken (over 1000 nodes)!" << std::endl;
                return false;
            }

            if (e.valid == false)
            {
                std::cout << "valid broken" << std::endl;
                return false;
            }

            if (e.face != f)
            {
                std::cout << "Not all half edges reference the same face" << std::endl;
                return false;
            }
        }

        if (count != 2)
        {
            std::cout << "Not a triangle mesh" << std::endl;
            return false;
        }
    }

    // check vertice edges
    // if they are actual outgoing edges
    for (int i = 0; i < (int)vertices.size(); ++i)
    {
        HalfVertex v = vertices[i];
        if (!v.valid) continue;

        if (v.halfEdge == -1)
        {
            std::cout << "-1 halfedge on vertex" << std::endl;
            return false;
        }

        HalfEdge e = edgeList[v.halfEdge];

        while (e.nextHalfEdge != v.halfEdge)
        {
            e = edgeList[e.nextHalfEdge];
        }

        if (e.vertex != i)
        {
            std::cout << "Vertex edge is broken!" << std::endl;
            return false;
        }
    }

    return true;
}

template <typename vertex_t, typename index_t>
void HalfEdgeMesh<vertex_t, index_t>::halfEdgeCollapse(int he)
{
    if (he == -1 || he >= edgeList.size()) return;

    HalfEdge e = edgeList[he];

    if (!e.valid) return;

    // no border
    //    SAIGA_ASSERT(e.oppositeHalfEdge != -1);


    int removeVertex = e.vertex;
    int newVertex    = edgeList[e.prevHalfEdge].vertex;


    //    vertices[removeVertex].v.position += vec4(0,0.5,0,0);
    //    vertices[newVertex].v.position += vec4(0,0.5,0,0);


    int removedFace1 = e.face;
    int removedFace2 = (e.oppositeHalfEdge == -1) ? -1 : edgeList[e.oppositeHalfEdge].face;



    // ================================================================================

    // iterate over all triangles of the removed vertex
    // update the vertices of the "incostd::ming" edges
    int startHf   = vertices[removeVertex].halfEdge;
    int currentHf = startHf;

    while (true)
    {
        HalfEdge& current = edgeList[currentHf];

        if (current.oppositeHalfEdge == -1) break;

        // this he points towards the vertex
        HalfEdge& flip = edgeList[current.oppositeHalfEdge];
        SAIGA_ASSERT(flip.vertex == removeVertex);
        flip.vertex = newVertex;

        currentHf = flip.nextHalfEdge;

        if (currentHf == startHf) break;
    }

    if (currentHf != startHf)
    {
        // we need to iterate in the opposite direction
        startHf   = vertices[removeVertex].halfEdge;
        currentHf = startHf;

        while (true)
        {
            HalfEdge& current = edgeList[currentHf];
            HalfEdge& flip    = edgeList[current.prevHalfEdge];
            SAIGA_ASSERT(flip.vertex == removeVertex);
            flip.vertex = newVertex;

            if (flip.oppositeHalfEdge == -1) break;

            currentHf = flip.oppositeHalfEdge;

            if (currentHf == startHf) break;
        }
        //        SAIGA_ASSERT(currentHf == startHf);
    }


    int w1 = -1, w2 = -1;
    // ================================================================================

    // set the opposite of the edges connected to the removed triangle

    // upper triangle
    HalfEdge tmp = edgeList[e.nextHalfEdge];
    int o1       = tmp.oppositeHalfEdge;
    w1           = tmp.vertex;
    tmp          = edgeList[tmp.nextHalfEdge];
    int o2       = tmp.oppositeHalfEdge;

    if (o1 != -1) edgeList[o1].oppositeHalfEdge = o2;
    if (o2 != -1) edgeList[o2].oppositeHalfEdge = o1;

    if (o1 != -1) vertices[w1].halfEdge = o1;



    //    std::cout << "o1,o2 " << o1 << "," << o2<< std::endl;

    // other triangle
    if (e.oppositeHalfEdge != -1)
    {
        tmp    = edgeList[e.oppositeHalfEdge];
        tmp    = edgeList[tmp.nextHalfEdge];
        int o3 = tmp.oppositeHalfEdge;
        w2     = tmp.vertex;
        tmp    = edgeList[tmp.nextHalfEdge];
        int o4 = tmp.oppositeHalfEdge;

        // fix neighbours
        if (o3 != -1) edgeList[o3].oppositeHalfEdge = o4;
        if (o4 != -1) edgeList[o4].oppositeHalfEdge = o3;

        if (o3 != -1) vertices[w2].halfEdge = o3;


        //        std::cout << "o3,o4 " << o3 << "," << o4 << std::endl;
    }



    // remove faces

    startHf   = he;
    currentHf = startHf;
    do
    {
        // remove this half edge
        HalfEdge& e = edgeList[currentHf];
        e.valid     = false;
        currentHf   = e.nextHalfEdge;
    } while (currentHf != startHf);
    faces[removedFace1].valid = false;

    if (e.oppositeHalfEdge != -1)
    {
        startHf   = e.oppositeHalfEdge;
        currentHf = startHf;
        do
        {
            // remove this half edge
            HalfEdge& e = edgeList[currentHf];
            e.valid     = false;
            currentHf   = e.nextHalfEdge;
        } while (currentHf != startHf);
        faces[removedFace2].valid = false;
    }

    //    removeFace(removedFace1);
    //    removeFace(removedFace2);

    vertices[removeVertex].valid = false;
    //    if(edgeList[vertices[w1].halfEdge].face == removedFace1)
    //    {
    //    }
}


template <typename vertex_t, typename index_t>
void HalfEdgeMesh<vertex_t, index_t>::flipEdge(int he)
{
    if (he == -1 || he >= edgeList.size()) return;

    HalfEdge e = edgeList[he];

    if (!e.valid) return;

    // border edges are not flipable
    if (e.oppositeHalfEdge == -1) return;

    // http://15462.courses.cs.cmu.edu/fall2015content/misc/HalfedgeEdgeOpImplementationGuide.pdf

    int h0 = he;
    int h1 = edgeList[h0].nextHalfEdge;
    int h2 = edgeList[h1].nextHalfEdge;

    int h3 = edgeList[h0].oppositeHalfEdge;
    int h4 = edgeList[h3].nextHalfEdge;
    int h5 = edgeList[h4].nextHalfEdge;

    int h6 = edgeList[h1].oppositeHalfEdge;
    int h7 = edgeList[h2].oppositeHalfEdge;
    int h8 = edgeList[h4].oppositeHalfEdge;
    int h9 = edgeList[h5].oppositeHalfEdge;

    int v0 = edgeList[h3].vertex;
    int v1 = edgeList[h0].vertex;
    int v2 = edgeList[h1].vertex;
    int v3 = edgeList[h4].vertex;

    // just a small check that these vertices are actual different
    if (v0 == v1 || v0 == v2 || v0 == v3 || v1 == v2 || v1 == v3 || v2 == v3)
    {
        std::cout << "broken mesh (non manifold). Not flipping..." << std::endl;
        return;
    }


    //    int f0 = edgeList[h0].face;
    //    int f1 = edgeList[h3].face;


    edgeList[h0].vertex = v2;
    edgeList[h1].vertex = v0;
    edgeList[h2].vertex = v3;

    edgeList[h3].vertex = v3;
    edgeList[h4].vertex = v1;
    edgeList[h5].vertex = v2;


    edgeList[h1].oppositeHalfEdge = h7;
    edgeList[h2].oppositeHalfEdge = h8;
    edgeList[h4].oppositeHalfEdge = h9;
    edgeList[h5].oppositeHalfEdge = h6;

    if (h7 != -1) edgeList[h7].oppositeHalfEdge = h1;
    if (h8 != -1) edgeList[h8].oppositeHalfEdge = h2;
    if (h9 != -1) edgeList[h9].oppositeHalfEdge = h4;
    if (h6 != -1) edgeList[h6].oppositeHalfEdge = h5;

    vertices[v0].halfEdge = h2;
    vertices[v1].halfEdge = h5;
    vertices[v2].halfEdge = h1;
    vertices[v3].halfEdge = h4;

    std::cout << h0 << "," << h1 << "," << h2 << "," << h3 << "," << h4 << "," << h5 << "," << std::endl;
    std::cout << h6 << "," << h7 << "," << h8 << "," << h9 << "," << std::endl;
    std::cout << std::endl;
}


template <typename vertex_t, typename index_t>
void HalfEdgeMesh<vertex_t, index_t>::removeFace(int f)
{
    if (f == -1) return;

    HalfFace& hf = faces[f];
    hf.valid     = false;


    // go in circle around face
    int startHf   = hf.halfEdge;
    int currentHf = startHf;
    do
    {
        // remove this half edge
        HalfEdge& e = edgeList[currentHf];
        e.valid     = false;

        if (e.oppositeHalfEdge != -1)
        {
            HalfEdge& op        = edgeList[e.oppositeHalfEdge];
            op.oppositeHalfEdge = -1;
        }

        currentHf = e.nextHalfEdge;
    } while (currentHf != startHf);
}

template <typename vertex_t, typename index_t>
void HalfEdgeMesh<vertex_t, index_t>::getNeighbours(int vertex, std::vector<int>& neighs)
{
    // ================================================================================

    // iterate over all triangles of the removed vertex
    // update the vertices of the "incostd::ming" edges
    int startHf   = vertices[vertex].halfEdge;
    int currentHf = startHf;

    while (true)
    {
        HalfEdge& current = edgeList[currentHf];
        neighs.push_back(current.vertex);

        if (current.oppositeHalfEdge == -1)
        {
            break;
        }

        // this he points towards the vertex
        HalfEdge& flip = edgeList[current.oppositeHalfEdge];
        currentHf      = flip.nextHalfEdge;

        if (currentHf == startHf)
        {
            //            neighs.push_back(edgeList[currentHf].vertex);
            return;
        }
    }

    //    if(currentHf != startHf)
    {
        // we need to iterate in the opposite direction
        startHf   = vertices[vertex].halfEdge;
        currentHf = startHf;

        while (true)
        {
            HalfEdge& current = edgeList[currentHf];
            HalfEdge& flip    = edgeList[current.prevHalfEdge];
            //            SAIGA_ASSERT(flip.vertex == removeVertex);
            //            flip.vertex = newVertex;

            if (flip.oppositeHalfEdge == -1) break;

            currentHf = flip.oppositeHalfEdge;


            if (currentHf == startHf) break;

            neighs.push_back(edgeList[currentHf].vertex);
        }
        //        SAIGA_ASSERT(currentHf == startHf);
    }
}

}  // namespace Saiga
