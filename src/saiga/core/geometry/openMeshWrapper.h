#pragma once

#include <iostream>

#include "triangle_mesh.h"

#ifdef SAIGA_USE_OPENMESH


#    include <OpenMesh/Core/IO/MeshIO.hh>
#    include <OpenMesh/Core/Mesh/Traits.hh>

#    include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"

namespace Saiga
{
struct SaigaOpenMeshTraits : public OpenMesh::DefaultTraits
{
    // use float color instead of byte color
    //  typedef OpenMesh::Vec3f Color;
    typedef OpenMesh::Vec3f Color;

    VertexAttributes(OpenMesh::Attributes::Color);
};


using OpenTriangleMesh = OpenMesh::TriMesh_ArrayKernelT<SaigaOpenMeshTraits>;

/**
 * Converts a Saiga::TriangleMesh to an OpenMesh triangle mesh.
 * Additional addributes on the vertices are lost.
 */
template <typename vertex_t, typename index_t, typename MeshT>
void triangleMeshToOpenMesh(const TriangleMesh<vertex_t, index_t>& src, MeshT& dst)
{
    //    dst.request_vertex_colors();

    std::vector<typename MeshT::VertexHandle> handles(src.vertices.size());
    for (int i = 0; i < (int)src.vertices.size(); ++i)
    {
        auto p   = src.vertices[i].position;
        auto vit = dst.add_vertex(typename MeshT::Point(p[0], p[1], p[2]));

        //        vec3 c = src.vertices[i].color;
        //        dst.set_color(vit, typename MeshT::Color(c[0],c.y,c[2]));

        handles[i] = vit;
    }

    std::vector<typename MeshT::VertexHandle> face_vhandles(3);
    for (int i = 0; i < (int)src.faces.size(); ++i)
    {
        auto f = src.faces[i];
        //        face_vhandles[0] = typename MeshT::VertexHandle(f.v1);
        //        face_vhandles[1] = typename MeshT::VertexHandle(f.v2);
        //        face_vhandles[2] = typename MeshT::VertexHandle(f.v3);

        face_vhandles[0] = handles[f.v1];
        face_vhandles[1] = handles[f.v2];
        face_vhandles[2] = handles[f.v3];
        dst.add_face(face_vhandles);
    }
}

template <typename vertex_t, typename index_t, typename MeshT>
void copyVertexColor(const TriangleMesh<vertex_t, index_t>& src, MeshT& dst)
{
    dst.request_vertex_colors();
    //    int i = 0;
    for (int i = 0; i < (int)src.vertices.size(); ++i)
    {
        auto c = src.vertices[i].color;
        dst.set_color(typename MeshT::VertexHandle(i), typename MeshT::Color(c[0], c[1], c[2]));

        //        vertex_t& ve = dst.vertices[i];
        //        typename MeshT::Color c = src.color( *v_it );
        //        ve.color = vec4(c[0],c[1],c[2],1);
    }
}

/**
 * Converts an OpenMesh triangle mesh to a Saiga::TriangleMesh
 * Additional addributes on the vertices are lost.
 */
template <typename vertex_t, typename index_t, typename MeshT>
void openMeshToTriangleMesh(const MeshT& src, TriangleMesh<vertex_t, index_t>& dst)
{
    dst.vertices.clear();
    dst.faces.clear();
    for (auto v_it = src.vertices_begin(); v_it != src.vertices_end(); ++v_it)
    {
        typename MeshT::Point v = src.point(*v_it);
        vertex_t ve;
        ve.position = vec4(v[0], v[1], v[2], 1);

        //        if(src.has_vertex_colors())
        //        {
        //            typename MeshT::Color c = src.color( *v_it );
        //            ve.color = vec4(c[0],c[1],c[2],1);
        //        }
        dst.vertices.push_back(ve);
    }
    std::vector<index_t> a;
    for (auto f_it = src.faces_begin(); f_it != src.faces_end(); ++f_it)
    {
        a.clear();
        for (auto fv_it = src.cfv_iter(*f_it); fv_it.is_valid(); ++fv_it)
        {
            auto vh = *fv_it;
            a.push_back(vh.idx());
        }
        SAIGA_ASSERT(a.size() == 3);
        dst.addFace(a.data());
    }
}

template <typename vertex_t, typename index_t, typename MeshT>
void copyVertexColor(const MeshT& src, TriangleMesh<vertex_t, index_t>& dst)
{
    SAIGA_ASSERT(src.has_vertex_colors());
    int i = 0;
    for (auto v_it = src.vertices_begin(); v_it != src.vertices_end(); ++v_it, ++i)
    {
        vertex_t& ve            = dst.vertices[i];
        typename MeshT::Color c = src.color(*v_it);
        ve.color                = vec4(c[0], c[1], c[2], 1);
    }
}

template <typename MeshT>
void saveOpenMesh(const MeshT& src, const std::string& file)
{
    try
    {
        if (!OpenMesh::IO::write_mesh(src, file))
        {
            std::cerr << "Cannot write mesh to file " << file << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
}


template <typename MeshT>
void loadOpenMesh(MeshT& src, const std::string& file)
{
    try
    {
        if (!OpenMesh::IO::read_mesh(src, file))
        {
            std::cerr << "Cannot read mesh to file " << file << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
}

}  // namespace Saiga



#endif
