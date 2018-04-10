#pragma once

#include "saiga/geometry/triangle_mesh.h"


#ifdef SAIGA_USE_OPENMESH


#include <OpenMesh/Core/IO/MeshIO.hh>
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"

namespace Saiga {

using OpenTriangleMesh = OpenMesh::TriMesh_ArrayKernelT<>;

/**
 * Converts a Saiga::TriangleMesh to an OpenMesh triangle mesh.
 * Additional addributes on the vertices are lost.
 */
template<typename vertex_t, typename index_t, typename MeshT>
void triangleMeshToOpenMesh(const TriangleMesh<vertex_t,index_t>& src, MeshT& dst)
{
    std::vector<typename MeshT::VertexHandle> handles(src.vertices.size());
    for(int i = 0; i < (int)src.vertices.size();++i)
    {
        vec3 p = src.vertices[i].position;
        handles[i] = dst.add_vertex(typename MeshT::Point(p.x,p.y,p.z));
    }

    for(int i = 0; i < (int)src.faces.size();++i)
    {
        auto f = src.faces[i];

        std::vector<typename MeshT::VertexHandle> face_vhandles;
        face_vhandles.push_back(handles[f.v1]);
        face_vhandles.push_back(handles[f.v2]);
        face_vhandles.push_back(handles[f.v3]);
        dst.add_face(face_vhandles);
    }
}

/**
 * Converts an OpenMesh triangle mesh to a Saiga::TriangleMesh
 * Additional addributes on the vertices are lost.
 */
template<typename vertex_t, typename index_t, typename MeshT>
void openMeshToTriangleMesh(const MeshT& src, TriangleMesh<vertex_t,index_t>& dst)
{
    dst.vertices.clear();
    dst.faces.clear();
    for (auto v_it = src.vertices_begin(); v_it != src.vertices_end(); ++v_it)
    {
        typename MeshT::Point v = src.point( *v_it );
        vec3 p(v[0],v[1],v[2]);

        VertexNC ve;
        ve.position = vec4(p,1);
        ve.color = vec4(1,0,0,1);
        dst.vertices.push_back(ve);
    }


    for(auto f_it = src.faces_begin(); f_it != src.faces_end(); ++f_it) {
        //        auto f = src.face(*f_it);

        std::vector<GLuint> a;
        for (auto fv_it = src.cfv_iter(*f_it); fv_it.is_valid(); ++fv_it)
        {
            auto vh = *fv_it;
            a.push_back(vh.idx());
        }
        SAIGA_ASSERT(a.size() == 3);
        dst.addFace(a.data());
    }
}

template<typename MeshT>
void saveOpenMesh(const MeshT& src, const std::string& file)
{
    try
    {
        if ( !OpenMesh::IO::write_mesh(src, file) )
        {
            std::cerr << "Cannot write mesh to file " << file << std::endl;

        }
    }
    catch( std::exception& x )
    {
        std::cerr << x.what() << std::endl;
    }
}


template<typename MeshT>
void loadOpenMesh(MeshT& src, const std::string& file)
{
    try
    {
        if ( !OpenMesh::IO::read_mesh(src, file) )
        {
            std::cerr << "Cannot read mesh to file " << file << std::endl;

        }
    }
    catch( std::exception& x )
    {
        std::cerr << x.what() << std::endl;
    }
}

}




#endif
