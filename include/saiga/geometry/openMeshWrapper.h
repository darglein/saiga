#pragma once

#include "saiga/geometry/triangle_mesh.h"


#ifdef SAIGA_USE_OPENMESH


#include <OpenMesh/Core/IO/MeshIO.hh>
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include <OpenMesh/Core/Mesh/Traits.hh>

namespace Saiga {


struct SaigaOpenMeshTraits : public OpenMesh::DefaultTraits
{
    //use float color instead of byte color
//  typedef OpenMesh::Vec3f Color;
    typedef OpenMesh::Vec3f Color;

    VertexAttributes (
    OpenMesh::Attributes::Color);
};


using OpenTriangleMesh = OpenMesh::TriMesh_ArrayKernelT<SaigaOpenMeshTraits>;

/**
 * Converts a Saiga::TriangleMesh to an OpenMesh triangle mesh.
 * Additional addributes on the vertices are lost.
 */
template<typename vertex_t, typename index_t, typename MeshT>
void triangleMeshToOpenMesh(const TriangleMesh<vertex_t,index_t>& src, MeshT& dst)
{

    dst.request_vertex_colors();

    std::vector<typename MeshT::VertexHandle> handles(src.vertices.size());
    for(int i = 0; i < (int)src.vertices.size();++i)
    {
        vec3 p = src.vertices[i].position;
         auto vit = dst.add_vertex(typename MeshT::Point(p.x,p.y,p.z));


         vec3 c = src.vertices[i].color;
         dst.set_color(vit, typename MeshT::Color(c.x,c.y,c.z));


         typename MeshT::Color c2 = dst.color( vit );
         vec3 c3 = vec3(c2[0],c2[1],c2[2]);

         if(length(c3 - c) != 0)
         {
             cout << c << " " << c3 << endl;
         }

         handles[i] = vit;
    }

    std::vector<typename MeshT::VertexHandle> face_vhandles(3);
    for(int i = 0; i < (int)src.faces.size();++i)
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
//        vec3 p(v[0],v[1],v[2]);




        VertexNC ve;
        ve.position = vec4(v[0],v[1],v[2],1);

        if(src.has_vertex_colors())
        {
            typename MeshT::Color c = src.color( *v_it );
            ve.color = vec4(c[0],c[1],c[2],1);
        }


        dst.vertices.push_back(ve);
    }


    std::vector<GLuint> a;
    for(auto f_it = src.faces_begin(); f_it != src.faces_end(); ++f_it)
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
