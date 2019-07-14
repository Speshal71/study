#include "mywidget.h"

MyWidget::MyWidget(int count, double upRad, double lowRad, double height): isometric(false), cone(count, upRad, lowRad, height) {}

void MyWidget::keyPressEvent(QKeyEvent *event)
{
    MyMatrix3D<double> matrix;
    double angle = M_PI / 180 * 5;

    if (event->key() == Qt::Key_Down || event->key() == Qt::Key_Up) {
        if (event->key() == Qt::Key_Up) {
            angle = -angle;
        }
        matrix =  MyMatrix3D<double>{{1, 0         ,  0         },
                                     {0, cos(angle), -sin(angle)},
                                     {0, sin(angle),  cos(angle)}};
    } else if (event->key() == Qt::Key_Left || event->key() == Qt::Key_Right) {
        if (event->key() == Qt::Key_Right) {
            angle = -angle;
        }
         matrix =  MyMatrix3D<double>{{cos(angle), -sin(angle), 0},
                                      {sin(angle),  cos(angle), 0},
                                      {0         ,  0         , 1}};
    } else if (event->key() == Qt::Key_Z || event->key() == Qt::Key_X) {
        if (event->key() == Qt::Key_X) {
            angle = -angle;
        }
         matrix =  MyMatrix3D<double>{{cos(angle), 0, -sin(angle)},
                                      {0         , 1,  0         },
                                      {sin(angle), 0,  cos(angle)}};
    } else if (event->key() == Qt::Key_Plus) {
         matrix =  MyMatrix3D<double>{{1.5,  0  , 0 },
                                      {0  ,  1.5, 0 },
                                      {0  ,  0  , 1.5}};
    }  else if (event->key() == Qt::Key_Minus) {
         matrix =  MyMatrix3D<double>{{0.66,  0   , 0   },
                                      {0   ,  0.66, 0   },
                                      {0   ,  0   , 0.66}};
    } else {
        if (event->key() == Qt::Key_Space) {
            isometric = !isometric;
            cone.projection(isometric);
        }
        matrix =  MyMatrix3D<double>{{1, 0, 0},
                                     {0, 1, 0},
                                     {0, 0, 1}};
    }

    cone.transform(matrix);

    update();
}

void MyWidget::paintEvent(QPaintEvent *event)
{
    QPainter p(this);

    cone.draw(p);
}

void MyWidget::approximateBy(int _count)
{
    cone.rebuild(_count);

    update();
}


