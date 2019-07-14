#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QtWidgets>

class MyFunc
{
private:
    double param;

public:
    MyFunc(double a): param(a) {}

    double operator()(double phi)
    {
        return param / phi;
    }

    void changeParameter(double a)
    {
        param = a;
    }
};

class MyGraph: public QWidget
{
private:
    const int fontSize = 8;
    int xCenter, yCenter;
    int segmentWidth = 30;
    int segmentLen = 5;
    int step = 1;
    int angleFrom = 1;
    int angleTo = 360;
    int param = 1;
    MyFunc func;

    void drawAxis(QPainter &p);
    void drawFunc(QPainter &p);

protected:
    void resizeEvent(QResizeEvent*);
    void paintEvent(QPaintEvent* event);

public:
    MyGraph(QWidget *parent = nullptr);
    void changeAngleFrom(int newAngle);
    void changeAngleTo(int newAngle);
    void changeParameter(int newParam);
};

#endif // MYWIDGET_H
