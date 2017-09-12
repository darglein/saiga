/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/crash.h"
#include "saiga/eigen/eigen.h"
#include "saiga/eigen/lse.h"
#include "saiga/time/performanceMeasure.h"
#include <random>

using namespace Saiga;
using std::cout;
using std::endl;

static void printVectorInstructions(){
	cout << "Eigen Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << endl;

    std::cout << "defined EIGEN Macros:" << std::endl;

#ifdef EIGEN_NO_DEBUG
    std::cout << "EIGEN_NO_DEBUG" << std::endl;
#else
    std::cout << "EIGEN_DEBUG" << std::endl;
#endif

#ifdef EIGEN_VECTORIZE_FMA
    std::cout << "EIGEN_VECTORIZE_FMA" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE3
    std::cout << "EIGEN_VECTORIZE_SSE3" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSSE3
    std::cout << "EIGEN_VECTORIZE_SSSE3" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_1
    std::cout << "EIGEN_VECTORIZE_SSE4_1" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_2
    std::cout << "EIGEN_VECTORIZE_SSE4_2" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_AVX
    std::cout << "EIGEN_VECTORIZE_AVX" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_AVX2
    std::cout << "EIGEN_VECTORIZE_AVX2" << std::endl;
#endif

    std::cout << std::endl;

}

inline void empty(){
}

struct EmptyOp{
    void operator ()(){}
};

inline void multMatrixVector(const Eigen::MatrixXf& M, const Eigen::VectorXf& x, Eigen::VectorXf& y){
    y += M * x;
}

template<typename MatrixType>
void randomMatrix(MatrixType& M)
{
    std::mt19937 engine(345345);
    std::uniform_real_distribution<float> dist(-1,1);
    for(int i = 0; i < M.rows(); ++i){
        for(int j = 0; j < M.cols(); ++j){
            M(i,j) = dist(engine);
        }

    }
}

int main(int argc, char *argv[]) {

    printVectorInstructions();
    Eigen::setNbThreads(1);


    catchSegFaults();

    const int N = 10000;

    //    using MatrixType = Eigen::MatrixXf;
    //    using VectorType = Eigen::VectorXf;

    using MatrixType = Eigen::Matrix<float,-1,-1,Eigen::ColMajor>;
    using VectorType = Eigen::Matrix<float,-1,1>;

    MatrixType M(N,N);
    VectorType x(N);
    VectorType y(N);

    randomMatrix(M);
    randomMatrix(x);

    cout << "random check: " << M(0,0) << " == " << -0.571635 << endl;
    measureFunction("multMatrixVector",100,multMatrixVector,M,x,y);


    MatrixType Ms(200,200);
    VectorType xs(200);
    randomMatrix(Ms);
    measureFunction("solveNullspaceSVD",100,solveNullspaceSVD<MatrixType,VectorType>,Ms,xs);


    using MatrixType2 = Eigen::Matrix<float,100,100,Eigen::ColMajor>;
    using VectorType2 = Eigen::Matrix<float,100,1>;
    MatrixType2 Ms2;
    VectorType2 xs2;
    randomMatrix(Ms2);
    measureFunction("solveNullspaceSVD2",100,solveNullspaceSVD<MatrixType2,VectorType2>,Ms2,xs2);
}
