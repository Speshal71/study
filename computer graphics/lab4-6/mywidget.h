#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_2_0>
#include <QKeyEvent>
#include <QTimer>

#include "matrix.h"

class MyWidget: public QOpenGLWidget, protected QOpenGLFunctions_2_0
{
public:
    GLfloat transformMatrix[16];
    bool rotation = false;

    int count;
    const double upRad;
    const double lowRad;
    const double height;

    bool isAnimated = false;
    QBasicTimer animationTimer;


    MyWidget(int _count, double _upRad, double _lowRad, double _height);

    void approximateBy(int _count);
    void startAnimation();

protected:
    void keyPressEvent(QKeyEvent *event);
    void timerEvent(QTimerEvent * event);

private:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void draw();
    void animationStep();
};

#endif // MYWIDGET_H
