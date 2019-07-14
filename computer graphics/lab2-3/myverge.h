#ifndef MYVERGE_H
#define MYVERGE_H

#include <QtWidgets>
#include <QVector>

#include "matrix.h"
#include "lightconsts.h"

class MyVerge
{
private:
    QVector<MyVector3D<double>> points;
    double d;
    MyVector3D<double> n;

public:
    MyVerge();

    bool addPoint(MyVector3D<double> point);
    void navigate(MyVector3D<double> point);
    void transform(MyMatrix3D<double> matrix);
    bool isVisible(MyVector3D<double> point);
    void draw(QPainter &p, const MyMatrix3D<double> &projectionMatrix);
};

#endif // MYVERGE_H
