#include "mycone.h"

#include <cmath>

MyCone::MyCone(int _count, double _upRad, double _lowRad, double _height):
    count(_count >= 3 ? _count : 3), upRad(_upRad), lowRad(_lowRad), height(_height), isometric(false)
{
    rebuild(count);
}

void MyCone::rebuild(int _count)
{
    count = (_count >= 3 ? _count : 3);
    verges.clear();
    verges.resize(count + 2);

    MyVector3D<double> internalPoint{0, 0, 0};
    MyVector3D<double> prevUpperVertex{upRad ,  height / 2, 0};
    MyVector3D<double> prevLowerVertex{lowRad, -height / 2, 0};

    for (int i = 0; i < count; ++i) {
        verges[0].addPoint(prevUpperVertex); //upper verge
        verges[1].addPoint(prevLowerVertex); //lower verge
        verges[i + 2].addPoint(prevUpperVertex);
        verges[i + 2].addPoint(prevLowerVertex);

        double angle = 2 * M_PI * (i + 1) / count;
        MyVector3D<double> newUpperVertex{upRad * cos(angle),  height / 2, upRad * sin(angle)};
        MyVector3D<double> newLowerVertex{lowRad * cos(angle), -height / 2, lowRad * sin(angle)};

        verges[i + 2].addPoint(newLowerVertex);
        verges[i + 2].addPoint(newUpperVertex);

        verges[i + 2].navigate(internalPoint);

        prevUpperVertex = newUpperVertex;
        prevLowerVertex = newLowerVertex;
    }

    verges[0].navigate(internalPoint);
    verges[1].navigate(internalPoint);
}

void MyCone::projection(bool _isometric)
{
    isometric = _isometric;
}

void MyCone::transform(MyMatrix3D<double> const &matrix)
{
    for (int i = 0; i < verges.size(); ++i) {
        verges[i].transform(matrix);
    }
}

void MyCone::draw(QPainter &p)
{
    for (int i = 0; i < verges.size(); ++i) {
        if (isometric) {
            if (verges[i].isVisible(MyVector3D<double>{300000,-300000,300000})) {
                verges[i].draw(p, isometricMatrix);
            }
        } else {
            if (verges[i].isVisible(MyVector3D<double>{0,0,300000})) {
                verges[i].draw(p, ortographicMatrix);
            }
        }
    }
}
