/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/geometry/half_edge_mesh.h"
#include "saiga/core/geometry/openMeshWrapper.h"
#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/objModelLoader.h"
#include "saiga/core/time/Time"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/random.h"
#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/VisionIncludes.h"

#include "ArapProblem.h"
#include "CeresArap.h"

#include <fstream>

using namespace Saiga;



template <typename vertex_t, typename index_t>
void saveMesh(const TriangleMesh<vertex_t, index_t>& mesh, const std::string& file)
{
    std::ofstream strm(file);

    strm << "OFF" << endl;
    // first line: number of vertices, number of faces, number of edges (can be ignored)
    strm << mesh.vertices.size() << " " << mesh.faces.size() << " 0" << endl;

    for (auto& v : mesh.vertices)
    {
        strm << v.position[0] << " " << v.position[1] << " " << v.position[2] << endl;
    }

    for (auto& f : mesh.faces)
    {
        strm << "3"
             << " " << f[0] << " " << f[1] << " " << f[2] << endl;
    }
}

int main(int, char**)
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    Saiga::Random::setSeed(93865023985);


    ObjModelLoader ol("bunny.obj");

    TriangleMesh<VertexNC, uint32_t> baseMesh;
    ol.toTriangleMesh(baseMesh);



    saveMesh(baseMesh, "test2.off");

    ArabMesh mesh;
    triangleMeshToOpenMesh(baseMesh, mesh);


    saveOpenMesh(mesh, "arab_0.off");
    ArapProblem problem;
    problem.createFromMesh(mesh);

    if (1)
    {
        int id = 0;
        // add an offset to the first vertex
        auto p = problem.vertices[id].translation();
        problem.target_indices.push_back(id);
        problem.target_positions.push_back(p + Vec3(0, 0.2, 0));
    }


    {
        ArapProblem cpy = problem;
        CeresArap ca;
        ca.optimize(cpy, 3);
    }

    problem.saveToMesh(mesh);
    saveOpenMesh(mesh, "arab_1.off");

    //    optimize(mesh);
    cout << "openmesh vertices: " << mesh.n_vertices() << endl;



    return 0;
}
