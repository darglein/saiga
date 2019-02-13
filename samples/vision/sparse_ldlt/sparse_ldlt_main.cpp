/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/random.h"
#include "saiga/vision/Random.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"

#include "Eigen/SparseCholesky"
using namespace Saiga;

class Sparse_LDLT_TEST
{
   public:
    using Block  = double;
    using Vector = double;

    using AType = Eigen::SparseMatrix<double>;
    using VType = Eigen::Matrix<double, -1, 1>;
    Sparse_LDLT_TEST(int n)
    {
        int nnzr = 1;
        A.resize(n, n);
        x.resize(n);
        b.resize(n);

        std::vector<Eigen::Triplet<Block> > trips;
        trips.reserve(nnzr * n * 2);

        for (int i = 0; i < n; ++i)
        {
            auto indices = Random::uniqueIndices(nnzr, n);
            std::sort(indices.begin(), indices.end());

            for (auto j : indices)
            {
                if (i != j)
                {
                    Block b = Random::sampleDouble(-1, 1);
                    trips.emplace_back(i, j, b);
                    trips.emplace_back(j, i, b);
                }
            }

            // Make sure we have a symmetric diagonal block
            Vector dv = Random::sampleDouble(-1, 1);
            Block D   = dv * dv;

            // Make sure the matrix is positiv
            D += 5;
            trips.emplace_back(i, i, D);
        }
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();


        Random::setRandom(x);
        Random::setRandom(b);

        cout << expand(A) << endl << endl;
        cout << b.transpose() << endl;
    }

    void solveDenseLDLT()
    {
        x = A.toDense().ldlt().solve(b);
        cout << "Dense LDLT error: " << (A * x - b).squaredNorm() << endl;
    }

    void solveSparseLDLT()
    {
        Eigen::SimplicialLDLT<AType, Eigen::Upper> ldlt;
        ldlt.compute(A);
        x = ldlt.solve(b);
        cout << "Sparse LDLT error: " << (A * x - b).squaredNorm() << endl;
        cout << ldlt.permutationP().toDenseMatrix() << endl;
    }

    AType A;
    VType x, b;
};


int main(int argc, char* argv[])
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();

    Random::setSeed(38670346483);

    Sparse_LDLT_TEST test(5);
    test.solveDenseLDLT();
    test.solveSparseLDLT();
    cout << "Done." << endl;
    return 0;
}
