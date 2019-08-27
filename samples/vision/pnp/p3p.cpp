
#include "p3p.h"

using std::cout;
using std::endl;


namespace Saiga
{
bool p3p::solve(double R[3][3], double t[3], double mu0, double mv0, double X0, double Y0, double Z0, double mu1,
                double mv1, double X1, double Y1, double Z1, double mu2, double mv2, double X2, double Y2, double Z2,
                double mu3, double mv3, double X3, double Y3, double Z3)
{
    double Rs[4][3][3], ts[4][3];

    int n = solve(Rs, ts, mu0, mv0, X0, Y0, Z0, mu1, mv1, X1, Y1, Z1, mu2, mv2, X2, Y2, Z2);

    if (n == 0) return false;

    int ns            = 0;
    double min_reproj = 0;
    for (int i = 0; i < n; i++)
    {
        double X3p    = Rs[i][0][0] * X3 + Rs[i][0][1] * Y3 + Rs[i][0][2] * Z3 + ts[i][0];
        double Y3p    = Rs[i][1][0] * X3 + Rs[i][1][1] * Y3 + Rs[i][1][2] * Z3 + ts[i][1];
        double Z3p    = Rs[i][2][0] * X3 + Rs[i][2][1] * Y3 + Rs[i][2][2] * Z3 + ts[i][2];
        double mu3p   = X3p / Z3p;
        double mv3p   = Y3p / Z3p;
        double reproj = (mu3p - mu3) * (mu3p - mu3) + (mv3p - mv3) * (mv3p - mv3);
        if (i == 0 || min_reproj > reproj)
        {
            ns         = i;
            min_reproj = reproj;
        }
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) R[i][j] = Rs[ns][i][j];
        t[i] = ts[ns][i];
    }

    return true;
}

int p3p::solve(double R[4][3][3], double t[4][3], double mu0, double mv0, double X0, double Y0, double Z0, double mu1,
               double mv1, double X1, double Y1, double Z1, double mu2, double mv2, double X2, double Y2, double Z2)
{
    double mk0, mk1, mk2;
    double norm;

    mu0  = mu0;
    mv0  = mv0;
    norm = sqrt(mu0 * mu0 + mv0 * mv0 + 1);
    mk0  = 1. / norm;
    mu0 *= mk0;
    mv0 *= mk0;

    mu1  = mu1;
    mv1  = mv1;
    norm = sqrt(mu1 * mu1 + mv1 * mv1 + 1);
    mk1  = 1. / norm;
    mu1 *= mk1;
    mv1 *= mk1;

    mu2  = mu2;
    mv2  = mv2;
    norm = sqrt(mu2 * mu2 + mv2 * mv2 + 1);
    mk2  = 1. / norm;
    mu2 *= mk2;
    mv2 *= mk2;

    double distances[3];
    distances[0] = sqrt((X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2) + (Z1 - Z2) * (Z1 - Z2));
    distances[1] = sqrt((X0 - X2) * (X0 - X2) + (Y0 - Y2) * (Y0 - Y2) + (Z0 - Z2) * (Z0 - Z2));
    distances[2] = sqrt((X0 - X1) * (X0 - X1) + (Y0 - Y1) * (Y0 - Y1) + (Z0 - Z1) * (Z0 - Z1));

    // Calculate angles
    double cosines[3];
    cosines[0] = mu1 * mu2 + mv1 * mv2 + mk1 * mk2;
    cosines[1] = mu0 * mu2 + mv0 * mv2 + mk0 * mk2;
    cosines[2] = mu0 * mu1 + mv0 * mv1 + mk0 * mk1;

    double lengths[4][3];
    int n = solve_for_lengths(lengths, distances, cosines);

    int nb_solutions = 0;
    for (int i = 0; i < n; i++)
    {
        double M_orig[3][3];

        M_orig[0][0] = lengths[i][0] * mu0;
        M_orig[0][1] = lengths[i][0] * mv0;
        M_orig[0][2] = lengths[i][0] * mk0;

        M_orig[1][0] = lengths[i][1] * mu1;
        M_orig[1][1] = lengths[i][1] * mv1;
        M_orig[1][2] = lengths[i][1] * mk1;

        M_orig[2][0] = lengths[i][2] * mu2;
        M_orig[2][1] = lengths[i][2] * mv2;
        M_orig[2][2] = lengths[i][2] * mk2;

        if (!align(M_orig, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2, R[nb_solutions], t[nb_solutions])) continue;

        nb_solutions++;
    }

    return nb_solutions;
}

int solve_deg2(double a, double b, double c, double& x1, double& x2)
{
    double delta = b * b - 4 * a * c;

    if (delta < 0) return 0;

    double inv_2a = 0.5 / a;

    if (delta == 0)
    {
        x1 = -b * inv_2a;
        x2 = x1;
        return 1;
    }

    double sqrt_delta = sqrt(delta);
    x1                = (-b + sqrt_delta) * inv_2a;
    x2                = (-b - sqrt_delta) * inv_2a;
    return 2;
}


/// Reference : Eric W. Weisstein. "Cubic Equation." From MathWorld--A Wolfram Web Resource.
/// http://mathworld.wolfram.com/CubicEquation.html
/// \return Number of real roots found.
int solve_deg3(double a, double b, double c, double d, double& x0, double& x1, double& x2)
{
    if (a == 0)
    {
        // Solve second order sytem
        if (b == 0)
        {
            // Solve first order system
            if (c == 0) return 0;

            x0 = -d / c;
            return 1;
        }

        x2 = 0;
        return solve_deg2(b, c, d, x0, x1);
    }

    // Calculate the normalized form x^3 + a2 * x^2 + a1 * x + a0 = 0
    double inv_a = 1. / a;
    double b_a = inv_a * b, b_a2 = b_a * b_a;
    double c_a = inv_a * c;
    double d_a = inv_a * d;

    // Solve the cubic equation
    double Q     = (3 * c_a - b_a2) / 9;
    double R     = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54;
    double Q3    = Q * Q * Q;
    double D     = Q3 + R * R;
    double b_a_3 = (1. / 3.) * b_a;

    if (Q == 0)
    {
        if (R == 0)
        {
            x0 = x1 = x2 = -b_a_3;
            return 3;
        }
        else
        {
            x0 = pow(2 * R, 1 / 3.0) - b_a_3;
            return 1;
        }
    }

    if (D <= 0)
    {
        // Three real roots
        double theta  = acos(R / sqrt(-Q3));
        double sqrt_Q = sqrt(-Q);
        x0            = 2 * sqrt_Q * cos(theta / 3.0) - b_a_3;
        x1            = 2 * sqrt_Q * cos((theta + 2 * M_PI) / 3.0) - b_a_3;
        x2            = 2 * sqrt_Q * cos((theta + 4 * M_PI) / 3.0) - b_a_3;

        return 3;
    }

    // D > 0, only one real root
    double AD = pow(fabs(R) + sqrt(D), 1.0 / 3.0) * (R > 0 ? 1 : (R < 0 ? -1 : 0));
    double BD = (AD == 0) ? 0 : -Q / AD;

    // Calculate the only real root
    x0 = AD + BD - b_a_3;

    return 1;
}

/// Reference : Eric W. Weisstein. "Quartic Equation." From MathWorld--A Wolfram Web Resource.
/// http://mathworld.wolfram.com/QuarticEquation.html
/// \return Number of real roots found.
int solve_deg4(double a, double b, double c, double d, double e, double& x0, double& x1, double& x2, double& x3)
{
    if (a == 0)
    {
        x3 = 0;
        return solve_deg3(b, c, d, e, x0, x1, x2);
    }

    // Normalize coefficients
    double inv_a = 1. / a;
    b *= inv_a;
    c *= inv_a;
    d *= inv_a;
    e *= inv_a;
    double b2 = b * b, bc = b * c, b3 = b2 * b;

    // Solve resultant cubic
    double r0, r1, r2;
    int n = solve_deg3(1, -c, d * b - 4 * e, 4 * c * e - d * d - b2 * e, r0, r1, r2);
    if (n == 0) return 0;

    // Calculate R^2
    double R2 = 0.25 * b2 - c + r0, R;
    if (R2 < 0) return 0;

    R            = sqrt(R2);
    double inv_R = 1. / R;

    int nb_real_roots = 0;

    // Calculate D^2 and E^2
    double D2, E2;
    if (R < 10E-12)
    {
        double temp = r0 * r0 - 4 * e;
        if (temp < 0)
            D2 = E2 = -1;
        else
        {
            double sqrt_temp = sqrt(temp);
            D2               = 0.75 * b2 - 2 * c + 2 * sqrt_temp;
            E2               = D2 - 4 * sqrt_temp;
        }
    }
    else
    {
        double u = 0.75 * b2 - 2 * c - R2, v = 0.25 * inv_R * (4 * bc - 8 * d - b3);
        D2 = u + v;
        E2 = u - v;
    }

    double b_4 = 0.25 * b, R_2 = 0.5 * R;
    if (D2 >= 0)
    {
        double D      = sqrt(D2);
        nb_real_roots = 2;
        double D_2    = 0.5 * D;
        x0            = R_2 + D_2 - b_4;
        x1            = x0 - D;
    }

    // Calculate E^2
    if (E2 >= 0)
    {
        double E   = sqrt(E2);
        double E_2 = 0.5 * E;
        if (nb_real_roots == 0)
        {
            x0            = -R_2 + E_2 - b_4;
            x1            = x0 - E;
            nb_real_roots = 2;
        }
        else
        {
            x2            = -R_2 + E_2 - b_4;
            x3            = x2 - E;
            nb_real_roots = 4;
        }
    }

    return nb_real_roots;
}


/// Given 3D distances between three points and cosines of 3 angles at the apex, calculates
/// the lentghs of the line segments connecting projection center (P) and the three 3D points (A, B, C).
/// Returned distances are for |PA|, |PB|, |PC| respectively.
/// Only the solution to the main branch.
/// Reference : X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang; "Complete Solution Classification for the
/// Perspective-Three-Point Problem" IEEE Trans. on PAMI, vol. 25, No. 8, August 2003 \param lengths3D Lengths of line
/// segments up to four solutions. \param dist3D Distance between 3D points in pairs |BC|, |AC|, |AB|. \param cosines
/// Cosine of the angles /_BPC, /_APC, /_APB. \returns Number of solutions. WARNING: NOT ALL THE DEGENERATE CASES ARE
/// IMPLEMENTED

int p3p::solve_for_lengths(double lengths[4][3], double distances[3], double cosines[3])
{
    double p = cosines[0] * 2;
    double q = cosines[1] * 2;
    double r = cosines[2] * 2;

    double inv_d22 = 1. / (distances[2] * distances[2]);
    double a       = inv_d22 * (distances[0] * distances[0]);
    double b       = inv_d22 * (distances[1] * distances[1]);

    double a2 = a * a, b2 = b * b, p2 = p * p, q2 = q * q, r2 = r * r;
    double pr = p * r, pqr = q * pr;

    // Check reality condition (the four points should not be coplanar)
    if (p2 + q2 + r2 - pqr - 1 == 0) return 0;

    double ab = a * b, a_2 = 2 * a;

    double A = -2 * b + b2 + a2 + 1 + ab * (2 - r2) - a_2;

    // Check reality condition
    if (A == 0) return 0;

    double a_4 = 4 * a;

    double B = q * (-2 * (ab + a2 + 1 - b) + r2 * ab + a_4) + pr * (b - b2 + ab);
    double C = q2 + b2 * (r2 + p2 - 2) - b * (p2 + pqr) - ab * (r2 + pqr) + (a2 - a_2) * (2 + q2) + 2;
    double D = pr * (ab - b2 + b) + q * ((p2 - 2) * b + 2 * (ab - a2) + a_4 - 2);
    double E = 1 + 2 * (b - a - ab) + b2 - b * p2 + a2;

    double temp = (p2 * (a - 1 + b) + r2 * (a - 1 - b) + pqr - a * pqr);
    double b0   = b * temp * temp;
    // Check reality condition
    if (b0 == 0) return 0;

    double real_roots[4];
    int n = solve_deg4(A, B, C, D, E, real_roots[0], real_roots[1], real_roots[2], real_roots[3]);

    if (n == 0) return 0;

    int nb_solutions = 0;
    double r3 = r2 * r, pr2 = p * r2, r3q = r3 * q;
    double inv_b0 = 1. / b0;

    // For each solution of x
    for (int i = 0; i < n; i++)
    {
        double x = real_roots[i];

        // Check reality condition
        if (x <= 0) continue;

        double x2 = x * x;

        double b1 =
            ((1 - a - b) * x2 + (q * a - q) * x + 1 - a + b) *
            (((r3 * (a2 + ab * (2 - r2) - a_2 + b2 - 2 * b + 1)) * x +

              (r3q * (2 * (b - a2) + a_4 + ab * (r2 - 2) - 2) +
               pr2 * (1 + a2 + 2 * (ab - a - b) + r2 * (b - b2) + b2))) *
                 x2 +

             (r3 * (q2 * (1 - 2 * a + a2) + r2 * (b2 - ab) - a_4 + 2 * (a2 - b2) + 2) +
              r * p2 * (b2 + 2 * (ab - b - a) + 1 + a2) + pr2 * q * (a_4 + 2 * (b - ab - a2) - 2 - r2 * b)) *
                 x +

             2 * r3q * (a_2 - b - a2 + ab - 1) + pr2 * (q2 - a_4 + 2 * (a2 - b2) + r2 * b + q2 * (a2 - a_2) + 2) +
             p2 * (p * (2 * (ab - a - b) + a2 + b2 + 1) + 2 * q * r * (b + a_2 - a2 - ab - 1)));

        // Check reality condition
        if (b1 <= 0) continue;

        double y = inv_b0 * b1;
        double v = x2 + y * y - x * y * r;

        if (v <= 0) continue;

        double Z = distances[2] / sqrt(v);
        double X = x * Z;
        double Y = y * Z;

        lengths[nb_solutions][0] = X;
        lengths[nb_solutions][1] = Y;
        lengths[nb_solutions][2] = Z;

        nb_solutions++;
    }

    return nb_solutions;
}

bool p3p::align(double M_end[3][3], double X0, double Y0, double Z0, double X1, double Y1, double Z1, double X2,
                double Y2, double Z2, double R[3][3], double T[3])
{
    // Centroids:
    double C_start[3], C_end[3];
    for (int i = 0; i < 3; i++) C_end[i] = (M_end[0][i] + M_end[1][i] + M_end[2][i]) / 3;
    C_start[0] = (X0 + X1 + X2) / 3;
    C_start[1] = (Y0 + Y1 + Y2) / 3;
    C_start[2] = (Z0 + Z1 + Z2) / 3;

    // Covariance matrix s:
    double s[3 * 3];
    for (int j = 0; j < 3; j++)
    {
        s[0 * 3 + j] = (X0 * M_end[0][j] + X1 * M_end[1][j] + X2 * M_end[2][j]) / 3 - C_end[j] * C_start[0];
        s[1 * 3 + j] = (Y0 * M_end[0][j] + Y1 * M_end[1][j] + Y2 * M_end[2][j]) / 3 - C_end[j] * C_start[1];
        s[2 * 3 + j] = (Z0 * M_end[0][j] + Z1 * M_end[1][j] + Z2 * M_end[2][j]) / 3 - C_end[j] * C_start[2];
    }

    double Qs[16], evs[4], U[16];

    Qs[0 * 4 + 0] = s[0 * 3 + 0] + s[1 * 3 + 1] + s[2 * 3 + 2];
    Qs[1 * 4 + 1] = s[0 * 3 + 0] - s[1 * 3 + 1] - s[2 * 3 + 2];
    Qs[2 * 4 + 2] = s[1 * 3 + 1] - s[2 * 3 + 2] - s[0 * 3 + 0];
    Qs[3 * 4 + 3] = s[2 * 3 + 2] - s[0 * 3 + 0] - s[1 * 3 + 1];

    Qs[1 * 4 + 0] = Qs[0 * 4 + 1] = s[1 * 3 + 2] - s[2 * 3 + 1];
    Qs[2 * 4 + 0] = Qs[0 * 4 + 2] = s[2 * 3 + 0] - s[0 * 3 + 2];
    Qs[3 * 4 + 0] = Qs[0 * 4 + 3] = s[0 * 3 + 1] - s[1 * 3 + 0];
    Qs[2 * 4 + 1] = Qs[1 * 4 + 2] = s[1 * 3 + 0] + s[0 * 3 + 1];
    Qs[3 * 4 + 1] = Qs[1 * 4 + 3] = s[2 * 3 + 0] + s[0 * 3 + 2];
    Qs[3 * 4 + 2] = Qs[2 * 4 + 3] = s[2 * 3 + 1] + s[1 * 3 + 2];



    jacobi_4x4(Qs, evs, U);


    //    exit(1);

    // Looking for the largest eigen value:
    int i_ev      = 0;
    double ev_max = evs[i_ev];
    for (int i = 1; i < 4; i++)
        if (evs[i] > ev_max) ev_max = evs[i_ev = i];

    // Quaternion:
    double q[4];
    for (int i = 0; i < 4; i++) q[i] = U[i * 4 + i_ev];

    double q02 = q[0] * q[0], q12 = q[1] * q[1], q22 = q[2] * q[2], q32 = q[3] * q[3];
    double q0_1 = q[0] * q[1], q0_2 = q[0] * q[2], q0_3 = q[0] * q[3];
    double q1_2 = q[1] * q[2], q1_3 = q[1] * q[3];
    double q2_3 = q[2] * q[3];

    R[0][0] = q02 + q12 - q22 - q32;
    R[0][1] = 2. * (q1_2 - q0_3);
    R[0][2] = 2. * (q1_3 + q0_2);

    R[1][0] = 2. * (q1_2 + q0_3);
    R[1][1] = q02 + q22 - q12 - q32;
    R[1][2] = 2. * (q2_3 - q0_1);

    R[2][0] = 2. * (q1_3 - q0_2);
    R[2][1] = 2. * (q2_3 + q0_1);
    R[2][2] = q02 + q32 - q12 - q22;

    for (int i = 0; i < 3; i++) T[i] = C_end[i] - (R[i][0] * C_start[0] + R[i][1] * C_start[1] + R[i][2] * C_start[2]);

    return true;
}

bool p3p::jacobi_4x4(double* A, double* D, double* U)
{
#if 1
    using MatrixType = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
    Eigen::Map<MatrixType> eA(A);
    Eigen::Map<Eigen::Matrix<double, 4, 1>> eD(D);
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> eu(U);


    // note: A is symmetric and real so we can use the self adjoint eigen solver.
    // The resulting eigen values are all real!
    // this is faster than the jacobi solver code below
    Eigen::SelfAdjointEigenSolver<MatrixType> eig(eA);


    for (int i = 0; i < 4; ++i)
    {
        D[i]   = eig.eigenvalues()(i);
        auto c = eig.eigenvectors().col(i);
        for (int j = 0; j < 4; ++j)
        {
            eu(j, i) = c(j);
        }
    }
    return true;
#else

    double B[4], Z[4];
    double Id[16] = {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};

    memcpy(U, Id, 16 * sizeof(double));

    B[0] = A[0];
    B[1] = A[5];
    B[2] = A[10];
    B[3] = A[15];
    memcpy(D, B, 4 * sizeof(double));
    memset(Z, 0, 4 * sizeof(double));

    for (int iter = 0; iter < 50; iter++)
    {
        double sum = fabs(A[1]) + fabs(A[2]) + fabs(A[3]) + fabs(A[6]) + fabs(A[7]) + fabs(A[11]);

        if (sum == 0.0) return true;

        double tresh = (iter < 3) ? 0.2 * sum / 16. : 0.0;
        for (int i = 0; i < 3; i++)
        {
            double* pAij = A + 5 * i + 1;
            for (int j = i + 1; j < 4; j++)
            {
                double Aij         = *pAij;
                double eps_machine = 100.0 * fabs(Aij);

                if (iter > 3 && fabs(D[i]) + eps_machine == fabs(D[i]) && fabs(D[j]) + eps_machine == fabs(D[j]))
                    *pAij = 0.0;
                else if (fabs(Aij) > tresh)
                {
                    double hh = D[j] - D[i], t;
                    if (fabs(hh) + eps_machine == fabs(hh))
                        t = Aij / hh;
                    else
                    {
                        double theta = 0.5 * hh / Aij;
                        t            = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if (theta < 0.0) t = -t;
                    }

                    hh = t * Aij;
                    Z[i] -= hh;
                    Z[j] += hh;
                    D[i] -= hh;
                    D[j] += hh;
                    *pAij = 0.0;

                    double c   = 1.0 / sqrt(1 + t * t);
                    double s   = t * c;
                    double tau = s / (1.0 + c);
                    for (int k = 0; k <= i - 1; k++)
                    {
                        double g = A[k * 4 + i], h = A[k * 4 + j];
                        A[k * 4 + i] = g - s * (h + g * tau);
                        A[k * 4 + j] = h + s * (g - h * tau);
                    }
                    for (int k = i + 1; k <= j - 1; k++)
                    {
                        double g = A[i * 4 + k], h = A[k * 4 + j];
                        A[i * 4 + k] = g - s * (h + g * tau);
                        A[k * 4 + j] = h + s * (g - h * tau);
                    }
                    for (int k = j + 1; k < 4; k++)
                    {
                        double g = A[i * 4 + k], h = A[j * 4 + k];
                        A[i * 4 + k] = g - s * (h + g * tau);
                        A[j * 4 + k] = h + s * (g - h * tau);
                    }
                    for (int k = 0; k < 4; k++)
                    {
                        double g = U[k * 4 + i], h = U[k * 4 + j];
                        U[k * 4 + i] = g - s * (h + g * tau);
                        U[k * 4 + j] = h + s * (g - h * tau);
                    }
                }
                pAij++;
            }
        }

        for (int i = 0; i < 4; i++) B[i] += Z[i];
        memcpy(D, B, 4 * sizeof(double));
        memset(Z, 0, 4 * sizeof(double));
    }

    return false;
#endif
}

}  // namespace Saiga
