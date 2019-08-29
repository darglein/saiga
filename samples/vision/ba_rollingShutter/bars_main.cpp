#include "saiga/core/framework/framework.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/ceres/CeresBARS.h"
#include "saiga/vision/ceres/CeresBASmooth.h"
#include "saiga/vision/util/Random.h"

using namespace Saiga;

// make sure se3 tangents are actual linear so they can be used like normal velocities
void checkLinearTangent()
{
    SE3 a;
    a.translation() = Vec3::Random();

    Vec6 t;
    t.segment<3>(0) = Vec3::Random();

    std::cout << a << std::endl;
    std::cout << t.transpose() << std::endl;
    std::cout << SE3::exp(t) << std::endl;
    std::cout << SE3::exp(t) * a << std::endl;
}


int main(const int argc, const char* argv[])
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(93865023985);



    Scene scene;
    scene.load(SearchPathes::data("vision/scene_gba.scene"));

    std::cout << scene << std::endl;

    OptimizationOptions oop;
    oop.debugOutput = true;

    //    CeresBARS cba;
    CeresBASmooth cba;
    cba.optimizationOptions = oop;
    cba.create(scene);
    cba.initAndSolve();
    std::cout << scene << std::endl;

    for (int i = 0; i < scene.extrinsics.size() - 2; ++i)
    {
        SmoothConstraint sc;
        sc.ex1    = i;
        sc.ex2    = i + 1;
        sc.ex3    = i + 2;
        sc.weight = 40;

        scene.smoothnessConstraints.push_back(sc);
    }


    cba.initAndSolve();

    std::cout << scene << std::endl;

    scene.save("gba_smooth.scene");

    return 0;
}
