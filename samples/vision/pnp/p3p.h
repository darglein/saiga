#ifndef P3P_H
#define P3P_H

#include "saiga/vision/VisionIncludes.h"

namespace Saiga
{
class p3p
{
   public:
    bool solve(const Vec3* worldPoints, const Vec2* normalizedPoints, SE3& result)
    {
        double rotation_matrix[3][3], translation_matrix[3];
        auto res = solve(rotation_matrix, translation_matrix, normalizedPoints[0](0), normalizedPoints[0](1),
                         worldPoints[0](0), worldPoints[0](1), worldPoints[0](2), normalizedPoints[1](0),
                         normalizedPoints[1](1), worldPoints[1](0), worldPoints[1](1), worldPoints[1](2),
                         normalizedPoints[2](0), normalizedPoints[2](1), worldPoints[2](0), worldPoints[2](1),
                         worldPoints[2](2), normalizedPoints[3](0), normalizedPoints[3](1), worldPoints[3](0),
                         worldPoints[3](1), worldPoints[3](2));
        if (!res) return res;
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(&rotation_matrix[0][0]);
        Vec3 t = Vec3(translation_matrix[0], translation_matrix[1], translation_matrix[2]);

        Quat q(R);
        result = SE3(q, t);
        return res;
    }


   private:
    int solve(double R[4][3][3], double t[4][3], double mu0, double mv0, double X0, double Y0, double Z0, double mu1,
              double mv1, double X1, double Y1, double Z1, double mu2, double mv2, double X2, double Y2, double Z2);
    bool solve(double R[3][3], double t[3], double mu0, double mv0, double X0, double Y0, double Z0, double mu1,
               double mv1, double X1, double Y1, double Z1, double mu2, double mv2, double X2, double Y2, double Z2,
               double mu3, double mv3, double X3, double Y3, double Z3);
    void init_inverse_parameters();
    int solve_for_lengths(double lengths[4][3], double distances[3], double cosines[3]);
    bool align(double M_start[3][3], double X0, double Y0, double Z0, double X1, double Y1, double Z1, double X2,
               double Y2, double Z2, double R[3][3], double T[3]);

    bool jacobi_4x4(double* A, double* D, double* U);
};
}  // namespace Saiga
#endif  // P3P_H
