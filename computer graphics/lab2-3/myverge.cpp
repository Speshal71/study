#include "myverge.h"
#include "QDebug"

MyVerge::MyVerge() {}

bool MyVerge::addPoint(MyVector3D<double> point) {
    points.push_back(point);

    if (points.size() == 3) {
        n = cross_product(MyVector3D<double>(points[0], points[1]), MyVector3D<double>(points[0], points[2]));
        d = -scolar_product(n, points[0]);
    }
    //check if point belongs to plane

    return true;
}

void MyVerge::navigate(MyVector3D<double> point)
{
    if ((scolar_product(n, point) + d) > 0) {
        for (int i = 0; i < 3; ++i) {
            n[i] = -n[i];
        }
        d = -d;
    }
}

void MyVerge::transform(MyMatrix3D<double> matrix)
{
    for (int i = 0; i < points.size(); ++i) {
        points[i] *= matrix;
    }

    n *= matrix;
    d = -scolar_product(n, points[0]);
}

bool MyVerge::isVisible(MyVector3D<double> point)
{
    if (points.size() >= 3 && (scolar_product(n, point) + d) > 0) {
        return true;
    }
    return false;
}

void MyVerge::draw(QPainter &p, const MyMatrix3D<double> &projectionMatrix)
{
    QPolygonF polygon;

    double k = 0; //light coefficient

    for (int i = 0; i < points.size(); ++i) {
        //creating of polygon to fill
        MyVector3D<double> temp = points[i] * projectionMatrix;
        polygon << QPointF(temp[0] + p.device()->width() / 2, -temp[1] + p.device()->height() / 2);

        //computation of light coefficient
        MyVector3D<double> lightVec(points[i], lightPos);
        k += scolar_product(n, lightVec) / n.length() / lightVec.length();
    }

    k = k * lightPower / points.size();
    k = (k >= 0 ? k : 0);

    QBrush brush;
    brush.setColor(QColor(255 * k, 0 * k, 0 * k));
    brush.setStyle(Qt::SolidPattern);

    QPainterPath path;
    path.addPolygon(polygon);

    p.fillPath(path, brush);
}
