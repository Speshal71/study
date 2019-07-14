#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

template<class T> using Matrix = std::vector<std::vector<T>>;
template<class T> using Vector = std::vector<T>;

Matrix<double> Identity(size_t N)
{
    Matrix<double> ret(N, Vector<double>(N, 0));

    for (size_t i = 0; i < N; ++i) {
        ret[i][i] = 1;
    }

    return ret;
}

double diagonalNorm(const Matrix<double> &m)
{
    double sum = 0;

    for (size_t i = 0; i < m.size(); ++i) {
        for (size_t j = i + 1; j < m.size(); ++j) {
            sum += m[i][j] * m[i][j];
        }
    }

    return sqrt(sum);
}

std::pair<size_t, size_t> getMaxAboveDiagonal(const Matrix<double> &m)
{
    double max = 0;
    std::pair<size_t, size_t> indexes;

    for (size_t i = 0; i < m.size(); ++i) {
        for (size_t j = i + 1; j < m.size(); ++j) {
            if (fabs(m[i][j]) > max) {
                max = fabs(m[i][j]);
                indexes = std::pair<size_t, size_t>(i, j);
            }
        }
    }

    return indexes;
}

size_t JacobiRotation(const Matrix<double> &m, Matrix<double> &U, Vector<double> &l, double precision)
{
    Matrix<double>  A = m;
    size_t iterCount = 0;

    U = Identity(A.size());
    l.resize(A.size());

    while (diagonalNorm(A) > precision) {
        std::pair<size_t, size_t> rotateAxis = getMaxAboveDiagonal(A);
        size_t i = rotateAxis.first;
        size_t j = rotateAxis.second;
        double angle;

        if (A[i][i] == A[j][j]) {
            angle = M_PI / 4;
        } else {
            angle = atan(2 * A[i][j] / (A[i][i] - A[j][j])) / 2;
        }

        double oldAi, oldAj, oldUi, oldUj;
        double sin_phi = sin(angle);
        double cos_phi = cos(angle);

        for (size_t k = 0; k < A.size(); ++k) {
            oldAi = A[i][k];
            oldAj = A[j][k];
            A[i][k] =  cos_phi * oldAi + sin_phi * oldAj;
            A[j][k] = -sin_phi * oldAi + cos_phi * oldAj;
        }
        
        for (size_t k = 0; k < A.size(); ++k) {
            oldAi = A[k][i];
            oldAj = A[k][j];
            A[k][i] =  cos_phi * oldAi + sin_phi * oldAj;
            A[k][j] = -sin_phi * oldAi + cos_phi * oldAj;

            oldUi = U[k][i];
            oldUj = U[k][j];
            U[k][i] =  cos_phi * oldUi + sin_phi * oldUj;
            U[k][j] = -sin_phi * oldUi + cos_phi * oldUj;
        }

        ++iterCount;
    }

    for (size_t i = 0; i < A.size(); ++i) {
        l[i] = A[i][i];
    }

    return iterCount;
}

int main()
{
    std::ifstream file("D:/myprog/numeric_methods/lab1/4/test.txt");

    if (file.is_open()) {
        size_t N;

        file >> N;

        Matrix<double> m = Identity(N);
        double precision;

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                file >> m[i][j];
            }
        }
        file >> precision;

        Matrix<double> U;
        Vector<double> l;
        size_t iterCount;

        iterCount = JacobiRotation(m, U, l, precision);

        for (size_t i = 0; i < l.size(); ++i) {
            std::cout << "Eigenvalue " << l[i] << " with the eigenvector:\n";
            for (size_t j = 0; j < U.size(); ++j) {
                std::cout << U[i][j] << '\n';
            }
            std::cout << "-------------------\n";
        }
        std::cout << "The total number of iteration is " << iterCount << " with the " << precision << " precision\n";
    } else {
        std::cout << "Can't open file!\n";
    }

    return 0;
}