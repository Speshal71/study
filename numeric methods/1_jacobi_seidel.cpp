#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

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

double normMat(const Matrix<double> &m)
{
    double norm = 0;

    for (size_t i = 0; i < m.size(); ++i) {
        double rowSum = 0;

        for (size_t j = 0; j < m[i].size(); ++j) {
            rowSum += fabs(m[i][j]);
        }

        if (rowSum > norm) {
            norm = rowSum;
        }
    }

    return norm;
}

double normVec(const Vector<double> &vec)
{
    double norm = 0;

    for(size_t i = 0; i < vec.size(); i++) {
        if (fabs(vec[i]) > norm) {
            norm = fabs(vec[i]);
        }
    }
    
    return norm;
}

Vector<double> operator-(Vector<double> v1, Vector<double> v2)
{
    Vector<double> ret;

    for(size_t i = 0; i < v1.size(); ++i) {
        ret.push_back(v1[i] - v2[i]);
    }
    
    return ret;
}

std::pair<size_t, std::vector<double>> Jacobi(const Matrix<double> &m, 
                                              const Vector<double> &b, 
                                              const double precision)
{
    size_t N = m.size();
    Vector<double> B(N);
    Matrix<double> A = Identity(N);

    for (size_t i = 0; i < N; ++i) {
        B[i] = b[i] / m[i][i];
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i][j] = -m[i][j] / m[i][i];
        }
        A[i][i] = 0;
    }

    double normA = normMat(A);
    Vector<double> X = B;
    Vector<double> newX(N);
    double eps = precision + 1;
    size_t iterCount = 0;

    while (eps > precision) {
        for (size_t i = 0; i < N; ++i) {
            newX[i] = 0;
            for (size_t j = 0; j < N; ++j) {
                newX[i] += A[i][j] * X[j];
            }
            newX[i] += B[i];
        }

        eps = normA / (1 - normA) * normVec(newX - X);
        ++iterCount;
        std::swap(newX, X);
    }

    return std::pair<size_t, Vector<double>>(iterCount, X);
}

std::pair<size_t, std::vector<double>> Seidel(const Matrix<double> &m, 
                                              const Vector<double> &b, 
                                              const double precision)
{
    size_t N = m.size();
    Vector<double> B(N);
    Matrix<double> A = Identity(N);

    for (size_t i = 0; i < N; ++i) {
        B[i] = b[i] / m[i][i];
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i][j] = -m[i][j] / m[i][i];
        }
        A[i][i] = 0;
    }

    double normA = normMat(A);
    Vector<double> X = B;
    Vector<double> newX(N);
    double eps = precision + 1;
    size_t iterCount = 0;

    while (eps > precision) {
        for (size_t i = 0; i < N; ++i) {
            newX[i] = 0;
            for (size_t j = 0; j < i; ++j) {
                newX[i] += A[i][j] * newX[j];
            }
            for (size_t j = i; j < N; ++j) {
                newX[i] += A[i][j] * X[j];
            }
            newX[i] += B[i];
        }

        eps = normA / (1 - normA) * normVec(newX - X);
        ++iterCount;
        std::swap(newX, X);
    }

    return std::pair<size_t, Vector<double>>(iterCount, X);
}

int main(int argc, char **argv)
{
    std::ifstream file("D:/myprog/numeric_methods/lab1/3/test.txt");

    if (!file.is_open()) {
        std::cout << "Can't open file!\n";
        return 0;
    }

    size_t N;

    file >> N;

    Matrix<double> m = Identity(N);
    Vector<double> b(N);
    double precision;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file >> m[i][j];
        }
    }

    for (int i = 0; i < N; ++i) {
        file >> b[i];
    }

    file >> precision;

    std::pair<size_t, std::vector<double>> solution = Jacobi(m, b, precision);

    std::cout << "Jacobi method:\n";
    std::cout << "\t precision = " << precision << '\n';
    std::cout << "\t number of iteration = " << solution.first << '\n';
    for (size_t i = 0; i < N; ++i) {
        std::cout << '\t' << solution.second[i] << '\n';
    }

    std::cout << '\n';

    solution = Seidel(m, b, precision);

    std::cout << "Seidel method:\n";
    std::cout << "\t precision = " << precision << '\n';
    std::cout << "\t number of iteration = " << solution.first << '\n';
    for (size_t i = 0; i < N; ++i) {
        std::cout << '\t' << solution.second[i] << '\n';
    }

    return 0;
}