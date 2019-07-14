#ifndef MATRIX_H
#define MATRIX_H

#include <QVector>
#include <QDebug>

const int dim = 3;

template <class T> class MyMatrix3D;

template <class T> class MyVector3D
{
private:
    QVector<T> v;

public:
    MyVector3D():  v(3, 0) {}

    MyVector3D(std::initializer_list<T> l): v(3, 0)
    {
        if (l.size() == dim) {
            v = l;
        }
    }

    MyVector3D(MyVector3D<T> from, MyVector3D<T> to): v(3)
    {
        for (int i = 0; i < dim; ++i) {
            v[i] = to[i] - from[i];
        }
    }

    T& operator[](int i) {
        return v[i];
    }

    const T& operator[](int i) const {
        return v[i];
    }

    MyVector3D<T>& operator=(MyVector3D<T> _v)
    {
        for (int i = 0; i < dim; ++i) {
            v[i] = _v[i];
        }

        return *this;
    }

    MyVector3D<T>& operator *=(const MyMatrix3D<T> &m)
    {
        QVector<T> temp(v);

        for (int i = 0; i < dim; ++i) {
            v[i] = 0;
            for (int j = 0; j < dim; ++j) {
                v[i] += m[i][j] * temp[j];
            }
        }

        return *this;
    }

    MyVector3D<T> operator *(const MyMatrix3D<T> &m)
    {
        MyVector3D<T> temp;

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                temp[i] += m[i][j] * v[j];
            }
        }

        return temp;
    }

    T length()
    {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
};

template <class T> class MyMatrix3D
{
private:
    QVector<MyVector3D<T>> m;

public:
    MyMatrix3D(): MyMatrix3D{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}} {}

    MyMatrix3D(std::initializer_list<MyVector3D<T>> _m): m(_m)
    {/*
        if (_m.size() == dim) {
            for (int i = 0; i < dim; ++i) {
                m[i] = _m[i];
            }
        }
        */
    }

    inline const MyVector3D<T>& operator[](int i) const {
        return m[i];
    }
};

template <class T> double scolar_product(MyVector3D<T> v1, MyVector3D<T> v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <class T> MyVector3D<T> cross_product(MyVector3D<T> v1, MyVector3D<T> v2)
{
    T x = v1[1]*v2[2]-v1[2]*v2[1];
    T y = v1[2]*v2[0]-v1[0]*v2[2];
    T z = v1[0]*v2[1]-v1[1]*v2[0];

    return MyVector3D<T>{x, y, z};
}


#endif // MATRIX_H
