#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QtWidgets>

class MyWidget: public QWidget
{
private:
    QVector<QRect> points;
    bool isMoving = false;
    int moveablePoint = -1;

    QPoint BSpline(double u);

protected:
    void paintEvent(QPaintEvent* event);
    void mousePressEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);

public:
    MyWidget();
};

#endif // MYWIDGET_H
