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

    catchSegFaults();

    const int N = 10000;


    Eigen::MatrixXf M = Eigen::MatrixXf::Random(N,N);
    Eigen::VectorXf x = Eigen::VectorXf::Random(N);
    Eigen::VectorXf y = Eigen::VectorXf::Random(N);

//    Eigen::Matrix<float,N,N> M = Eigen::Matrix<float,N,N>::Random();
//    Eigen::Matrix<float,N,1> x = Eigen::Matrix<float,N,1>::Random();
//    Eigen::Matrix<float,N,1> y = Eigen::Matrix<float,N,1>::Random();
//    Eigen::VectorXf x = Eigen::VectorXf::Random(N);
//    Eigen::VectorXf y = Eigen::VectorXf::Random(N);

//    {
//        ScopedTimerPrint tim("test");
//        y = M * x;
//    }

//    measureFunction("empty",10000,empty);
//    measureObject("empty",10000,EmptyOp());

    measureFunction("multMatrixVector",100,multMatrixVector,M,x,y);
//    measureObject("multMatrixVector",100,[&](){ y = M*x;});


    Eigen::MatrixXf Ms = Eigen::MatrixXf::Random(100,500);
    Eigen::VectorXf xs = Eigen::VectorXf::Random(100);
    measureFunction("solveNullspaceSVD",100,solveNullspaceSVD<Eigen::MatrixXf,Eigen::VectorXf>,Ms,xs);

//    std::cout << y.transpose() << std::endl;
}
