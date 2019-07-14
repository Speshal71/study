#ifndef MYCONE_H
#define MYCONE_H

#include <QtWidgets>
#include <QVector>
#include <vector>

#include "matrix.h"
#include "myverge.h"

const MyMatrix3D<double> isometricMatrix = {{0.7071,  0     , -0.7071},
                                            {0.4082,  0.8165,  0.4082},
                                            {0.5773, -0.5773,  0.5773}};

const MyMatrix3D<double> ortographicMatrix = {{1, 0, 0},
                                              {0, 1, 0},
                                              {0, 0, 1}};

class MyCone: public QWidget
{
private:
    std::vector<MyVerge> verges;
    bool isometric;
    int count;
    double upRad;
    double lowRad;
    double height;

public:
    MyCone(int _count, double _upRad, double _lowRad, double _height);

    void transform(MyMatrix3D<double> const &matrix);
    void draw(QPainter &p);
    void projection(bool _isometric);
    void rebuild(int _count);

};

#endif // MYCONE_H
