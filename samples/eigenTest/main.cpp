/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/crash.h"
#include "saiga/eigen/eigen.h"
#include "saiga/eigen/lse.h"
#include "saiga/time/performanceMeasure.h"

using namespace Saiga;
using std::cout;
using std::endl;

static void printVectorInstructions(){
    std::cout << "defined EIGEN Vector Instructions:" << std::endl;
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

}

inline void empty(){
}

struct EmptyOp{
    void operator ()(){}
};

inline void multMatrixVector(const Eigen::MatrixXf& M, const Eigen::VectorXf& x, Eigen::VectorXf& y){
    y += M * x;
}

int main(int argc, char *argv[]) {

    printVectorInstructions();
    Eigen::setNbThreads(1);
    srand(3649346);
    cout << "random: " << rand() << endl;

    catchSegFaults();

    const int N = 10000;


    Eigen::MatrixXf M = Eigen::MatrixXf::Random(N,N);
    Eigen::VectorXf x = Eigen::VectorXf::Random(N);
    Eigen::VectorXf y = Eigen::VectorXf::Random(N);

    measureFunction("multMatrixVector",100,multMatrixVector,M,x,y);


    using MatrixType = Eigen::MatrixXf;
    using VectorType = Eigen::VectorXf;
    MatrixType Ms = MatrixType::Random(200,200);
    VectorType xs = VectorType::Random(200);
    measureFunction("solveNullspaceSVD",100,solveNullspaceSVD<MatrixType,VectorType>,Ms,xs);


    using MatrixType2 = Eigen::Matrix<float,100,100>;
    using VectorType2 = Eigen::Matrix<float,100,1>;
    MatrixType2 Ms2 = MatrixType2::Random();
    VectorType2 xs2 = VectorType2::Random();
    measureFunction("solveNullspaceSVD2",100,solveNullspaceSVD<MatrixType2,VectorType2>,Ms2,xs2);
//    std::cout << y.transpose() << std::endl;
}
