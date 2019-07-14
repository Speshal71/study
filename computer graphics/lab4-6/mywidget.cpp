#include "mywidget.h"

#include <QDebug>

GLfloat rotVecX = 0;
GLfloat rotVecY = 0;
GLfloat rotVecZ = 0;

const GLfloat direction[] = {0.0f, 0.0f, 10.0f, 1.0f};
GLfloat intensity[] = {0.95f, 0.95f, 0.95f, 1.0f};
const GLfloat ambient_intensity[] = {0.05f, 0.05f, 0.05f, 1.0f};

int intensityStep = 0;

MyWidget::MyWidget(int _count, double _upRad, double _lowRad, double _height):
    count(_count >= 3 ? _count : 3), upRad(_upRad), lowRad(_lowRad), height(_height)
{

}

void MyWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0.0, 0.0, 0.0, 0.0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glTranslatef(0.0f, 0.0f, -15.0f);
    glGetFloatv(GL_MODELVIEW_MATRIX, transformMatrix);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_LIGHTING);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_intensity);

    glEnable(GL_LIGHT0);

    glLightfv(GL_LIGHT0, GL_POSITION, direction);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, intensity);

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
}

void MyWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    if (w <= h) {
        glOrtho(-1.5, 1.5, -1.5 * h / w, 1.5 * h / w, -20.0, 20.0);
    } else {
        glOrtho(-1.5 * w / h, 1.5 * w / h, -1.5, 1.5, -20.0, 20.0);
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void MyWidget::draw()
{
    glBegin(GL_QUAD_STRIP);

    for (int i = 0; i < (count + 1); ++i) {
        double angle = 2 * M_PI * i / count;
        MyVector3D<double> upperVer{upRad * cos(angle),  height / 2, upRad * sin(angle)};
        MyVector3D<double> lowerVer{lowRad * cos(angle), -height / 2, lowRad * sin(angle)};

        MyVector3D<double> a(upperVer);

        MyVector3D<double> b(upperVer, lowerVer);
        double coef = - scolar_product(a, b) / scolar_product(b, b);

        MyVector3D<double> n(a + b * coef);

        glNormal3f(n[0], n[1], n[2]);
        glVertex3f(upperVer[0], upperVer[1], upperVer[2]);
        glNormal3f(n[0], n[1], n[2]);
        glVertex3f(lowerVer[0], lowerVer[1], lowerVer[2]);
    }

    glEnd();


    glBegin(GL_POLYGON);

    glNormal3f(0, 1, 0);
    for (int i = 0; i < count; ++i) {
        double angle = 2 * M_PI * i / count;
        MyVector3D<double> upperVer{upRad * cos(angle),  height / 2, upRad * sin(angle)};

        glVertex3f(upperVer[0], upperVer[1], upperVer[2]);
    }

    glEnd();


    glBegin(GL_POLYGON);

    glNormal3f(0, -1, 0);
    for (int i = 0; i < count; ++i) {
        double angle = 2 * M_PI * i / count;
        MyVector3D<double> lowerVer{lowRad * cos(angle), -height / 2, lowRad * sin(angle)};

        glVertex3f(lowerVer[0], lowerVer[1], lowerVer[2]);
    }

    glEnd();
}

void MyWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glColor3f(1.0, 1.0, 1.0);

    glShadeModel(GL_FLAT);

    glLightfv(GL_LIGHT0, GL_DIFFUSE, intensity);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushMatrix();

    if (rotation) {
        glRotatef(5, rotVecX, rotVecY, rotVecZ);
        rotation = false;
    }
    glMultMatrixf(transformMatrix);
    glGetFloatv(GL_MODELVIEW_MATRIX, transformMatrix);

    glPopMatrix();

    glTranslatef(0.0f, 0.0f, -10.0f);
    glMultMatrixf(transformMatrix);

    draw();

}

void MyWidget::keyPressEvent(QKeyEvent *event)
{
    rotVecX = rotVecY = rotVecZ = 0;

    if (event->key() == Qt::Key_Down){
        rotVecX = 1;
    } else if (event->key() == Qt::Key_Up) {
        rotVecX = -1;
    } else if (event->key() == Qt::Key_Right) {
        rotVecY = 1;
    } else if (event->key() == Qt::Key_Left) {
        rotVecY = -1;
    } else if (event->key() == Qt::Key_Z) {
        rotVecZ = 1;
    } else if (event->key() == Qt::Key_X) {
        rotVecZ = -1;
    }

    rotation = true;

    update();
}

void MyWidget::approximateBy(int _count)
{
    count = (_count >= 3 ? _count : 3);

    update();
}

void MyWidget::startAnimation()
{
    if (isAnimated) {
        animationTimer.stop();

        for (int i = 0; i < 3; ++i) {
            intensity[i] = 0.95f;
        }

        update();
    } else {
        intensityStep = 25;
        animationTimer.start(50, this);
    }

    isAnimated = !isAnimated;
}

void MyWidget::timerEvent(QTimerEvent * event)
{
    if (event->timerId() != animationTimer.timerId()) {
        QOpenGLWidget::timerEvent(event);
        return;
    }

    animationStep();
}


void MyWidget::animationStep()
{
    for (int i = 0; i < 3; ++i) {
        intensity[i] = 0.95f * sin(M_PI * intensityStep / 50);
    }

    intensityStep = (intensityStep + 1) % 50;

    update();
}
