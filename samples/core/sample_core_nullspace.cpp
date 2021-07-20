/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"
#include "saiga/core/math/HomogeneousLSE.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/table.h"
using namespace Saiga;

template <int N>
class NullSpaceTest
{
    using T      = double;
    using Matrix = Eigen::Matrix<T, N, N>;
    using Vector = Eigen::Matrix<T, N, 1>;
    Matrix A;
    Vector x;
    int its = 1000;

   public:
    NullSpaceTest(double singularity = 0)
    {
        static_assert(N >= 3, "N must be larger than 3");

        // create a rank defficient matrix
        A.setRandom();

        // enforce it with svd
        // det(F)=0
        auto svde = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto svs  = svde.singularValues();
        // do not make it exactly 0 to test stability
        svs(A.cols() - 1) = singularity;

        Eigen::DiagonalMatrix<double, N> sigma;
        sigma.diagonal() = svs;
        A                = svde.matrixU() * sigma * svde.matrixV().transpose();
        x.setZero();

        {
            std::cout << "Running " << N << "x" << N << " test with " << its
                      << " iterations. Expected error: " << singularity << std::endl;
            Saiga::Table table({15, 15, 15});
            table << "Name"
                  << "Error"
                  << "Time (ms)";

            auto t = lu();
            table << "LU" << error() << t;
            t = qr();
            table << "QR" << error() << t;
            t = cod();
            table << "COD" << error() << t;
            t = jacobisvd();
            table << "Jacobi-SVD" << error() << t;
            t = bdcsvd();
            table << "BDC-SVD" << error() << t;
        }
        std::cout << std::endl;
    }



    auto lu()
    {
        auto stats = measureObject(its, [&]() { solveHomogeneousLU(A, x); });
        return stats.median;
    }

    auto qr()
    {
        auto stats = measureObject(its, [&]() { solveHomogeneousQR(A, x); });
        return stats.median;
    }

    auto cod()
    {
        auto stats = measureObject(its, [&]() { solveHomogeneousCOD(A, x); });
        return stats.median;
    }

    auto jacobisvd()
    {
        auto stats = measureObject(its, [&]() { solveHomogeneousJacobiSVD(A, x); });
        return stats.median;
    }

    auto bdcsvd()
    {
        auto stats = measureObject(its, [&]() { solveHomogeneousBDCSVD(A, x); });
        return stats.median;
    }

    auto error()
    {
        double e = (A * x).norm();

        if (x.norm() < 0.1) e = std::numeric_limits<double>::infinity();
        x.setZero();
        //        std::cout << x.transpose() << std::endl;
        return e;
        //        std::cout << "Error: " << e << std::endl;
    }
};


int main(int, char**)
{
    catchSegFaults();
    std::cout << "Testing different Eigen methods to find the nullspace of a matrix Ax=0, for x!=0" << std::endl;


    NullSpaceTest<4> nst1(0);
    NullSpaceTest<4> nst1_s(1e-2);

    NullSpaceTest<16> nst2(0);
    NullSpaceTest<16> nst2_s(1e-2);

    std::cout << "Done." << std::endl;

    return 0;
}
