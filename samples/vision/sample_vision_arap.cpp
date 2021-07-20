/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/geometry/half_edge_mesh.h"
#include "saiga/core/geometry/openMeshWrapper.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/model/model_loader_obj.h"
#include "saiga/core/model/model_loader_ply.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/arap/ArapProblem.h"
#include "saiga/vision/ceres/CeresArap.h"
#include "saiga/vision/g2o/G2OArap.h"
#include "saiga/vision/recursive/RecursiveArap.h"

#include <fstream>

using namespace Saiga;

std::vector<std::string> getFiles()
{
    std::vector<std::string> files;
    files.insert(files.end(), {"dragon_10k.ply"});
    files.insert(files.end(), {"dragon_25k.ply"});
    files.insert(files.end(), {"dragon_100k.ply"});
    files.insert(files.end(), {"dragon_250k.ply"});
    return files;
}


void test_to_file(const OptimizationOptions& options, const std::string& file, int its)
{
    std::cout << options << std::endl;
    std::cout << "Running long performance test to file..." << std::endl;

    auto files = getFiles();


    std::ofstream strm(file);
    strm << "file,vertices,constraints,targets,density,solver_type,iterations,time_recursive,time_ceres,time_g2o"
         << std::endl;


    Saiga::Table table({20, 20, 15, 15});

    for (auto file : files)
    {
        ArapProblem problem;

        if (hasEnding(file, ".ply"))
        {
            PLYLoader pl(file);

            TriangleMesh<VertexNC, uint32_t> baseMesh = pl.mesh;

            ArabMesh mesh;
            triangleMeshToOpenMesh(baseMesh, mesh);
            problem.createFromMesh(mesh);
        }

        {
            int id = 0;
            // add an offset to the first vertex
            auto p = problem.vertices[id].translation();
            problem.target_indices.push_back(id);
            problem.target_positions.push_back(p + Vec3(0, 0.02, 0));
        }
        {
            //            problem.createFromMesh(mesh);
            int id = 10;
            // add an offset to the first vertex
            auto p = problem.vertices[id].translation();
            problem.target_indices.push_back(id);
            problem.target_positions.push_back(p + Vec3(0, -0.02, 0));
        }
        std::vector<std::shared_ptr<ArapBase>> solvers;
        solvers.push_back(std::make_shared<RecursiveArap>());
        solvers.push_back(std::make_shared<CeresArap>());
        solvers.push_back(std::make_shared<G2OArap>());



#if 1

        std::cout << "> Initial Error: " << problem.chi2() << std::endl;
        table << "Name"
              << "Final Error"
              << "Time_LS"
              << "Time_Total";

        strm << file << "," << problem.vertices.size() << "," << problem.constraints.size() << ","
             << problem.target_indices.size() << "," << problem.density() << "," << (int)options.solverType << ","
             << (int)options.maxIterations;

        for (auto& s : solvers)
        {
            std::vector<double> times;
            std::vector<double> timesl;
            double chi2;
            for (int i = 0; i < its; ++i)
            {
                auto cpy = problem;
                s->create(cpy);
                auto opt                 = dynamic_cast<Optimizer*>(s.get());
                opt->optimizationOptions = options;
                SAIGA_ASSERT(opt);
                auto result = opt->initAndSolve();
                chi2        = cpy.chi2();
                times.push_back(result.total_time);
                timesl.push_back(result.linear_solver_time);
            }


            auto t  = Statistics(times).median / 1000.0 / options.maxIterations;
            auto tl = Statistics(timesl).median / 1000.0 / options.maxIterations;
            table << s->name << chi2 << tl << t;

            strm << "," << t;
        }
        strm << std::endl;
#endif
        std::cout << std::endl;
    }
}

int main(int, char**)
{
    initSaigaSampleNoWindow();

    Saiga::Random::setSeed(93865023985);

    OptimizationOptions options;
    options.solverType    = OptimizationOptions::SolverType::Direct;
    options.debugOutput   = false;
    options.maxIterations = 5;
    //    options.initialLambda = 100;

    options.maxIterativeIterations = 100;
    options.iterativeTolerance     = 0;
    test_to_file(options, "arab.csv", 1);


    //    GenericModel testModel("bunny.obj");
    //    GenericModel testModel2("dragon_10k.ply");
    return 0;

#if 0
    ObjModelLoader ol("bunny.obj");
    PLYLoader pl("dragon_100k.ply");

    TriangleMesh<VertexNC, uint32_t> baseMesh;
    //    ol.toTriangleMesh(baseMesh);

    baseMesh = pl.mesh;



    saveMesh(baseMesh, "test2.off");

    ArabMesh mesh;
    triangleMeshToOpenMesh(baseMesh, mesh);


    saveOpenMesh(mesh, "arab_0.off");
    ArapProblem problem;

    if (1)
    {
        problem.createFromMesh(mesh);
        {
            int id = 0;
            // add an offset to the first vertex
            auto p = problem.vertices[id].translation();
            problem.target_indices.push_back(id);
            problem.target_positions.push_back(p + Vec3(0, 0.2, 0));
        }
        {
            //            problem.createFromMesh(mesh);
            int id = 10;
            // add an offset to the first vertex
            auto p = problem.vertices[id].translation();
            problem.target_indices.push_back(id);
            problem.target_positions.push_back(p + Vec3(0, -0.2, 0));
        }
    }
    else
    {
        problem.makeTest();
    }



    {
        ArapProblem cpy = problem;
        CeresArap ca;
        ca.optimizationOptions = options;
        ca.create(cpy);
        auto res = ca.solve();
        std::cout << res << std::endl;
    }


    {
        ArapProblem cpy = problem;
        RecursiveArap ca;
        ca.arap                = &cpy;
        ca.optimizationOptions = options;
        auto res               = ca.solve();
        std::cout << res << std::endl;
    }

    //    problem.saveToMesh(mesh);
    //    saveOpenMesh(mesh, "arab_1.off");

    //    optimize(mesh);
    //    std::cout << "openmesh vertices: " << mesh.n_vertices() << std::endl;



    return 0;
#endif
}
