#include "mygraph.h"

MyGraph::MyGraph(QWidget *parent): QWidget(parent), func(param)
{
    xCenter = width() / 2;
    yCenter = height() / 2;
}

void MyGraph::resizeEvent(QResizeEvent* event)
{
   QWidget::resizeEvent(event);

   xCenter = width() / 2;
   yCenter = height() / 2;
}

void MyGraph::changeAngleFrom(int newAngle)
{
    angleFrom = newAngle;

    update();
}

void MyGraph::changeAngleTo(int newAngle)
{
    angleTo = newAngle;

    update();
}

void MyGraph::changeParameter(int newParam)
{
    func.changeParameter(newParam);

    update();
}

void MyGraph::drawAxis(QPainter &p)
{
    //drawing raw axes
    p.drawLine(QLine(0, yCenter, width(), yCenter));
    p.drawLine(QLine(xCenter, 0, xCenter, height()));
    p.drawText(QPoint(xCenter + segmentLen + 1, yCenter + segmentLen + fontSize + 1), "0");

    //marking x axis segments
    for (int i = segmentWidth, j = 1; i < (xCenter - segmentWidth); i += segmentWidth, j++) {
        int xSegment = xCenter + i;
        p.drawLine(QLine(xSegment, yCenter + segmentLen, xSegment, yCenter - segmentLen));
        p.drawText(QPoint(xSegment, yCenter + segmentLen + fontSize + 1), QString::number(j));
        xSegment = xCenter - i;
        p.drawLine(QLine(xSegment, yCenter + segmentLen, xSegment, yCenter - segmentLen));
        p.drawText(QPoint(xSegment, yCenter + segmentLen + fontSize + 1), QString::number(-j));
    }
    //drawing x axis arrow
    p.drawLine(QLine(width(), yCenter, width() - segmentWidth, yCenter + segmentLen));
    p.drawLine(QLine(width(), yCenter, width() - segmentWidth, yCenter - segmentLen));
    p.drawText(QPoint(width() - segmentWidth, yCenter - segmentLen - fontSize - 1), "X");

    //marking y axis segments
    for (int i = segmentWidth, j = 1; i < (yCenter - segmentWidth); i += segmentWidth, j++) {
        int ySegment = yCenter + i;
        p.drawLine(QLine(xCenter - segmentLen, ySegment, xCenter + segmentLen, ySegment));
        p.drawText(QPoint(xCenter + segmentLen + 1, ySegment), QString::number(-j));
        ySegment = yCenter - i;
        p.drawLine(QLine(xCenter - segmentLen, ySegment, xCenter + segmentLen, ySegment));
        p.drawText(QPoint(xCenter + segmentLen + 1, ySegment), QString::number(j));
    }
    //drawing y axis arrow
    p.drawLine(QLine(xCenter - segmentLen, segmentWidth, xCenter, 0));
    p.drawLine(QLine(xCenter + segmentLen, segmentWidth, xCenter, 0));
    p.drawText(QPoint(xCenter - segmentLen - fontSize - 1, segmentWidth), "Y");
}

void MyGraph::drawFunc(QPainter &p)
{
    QPainterPath path;

    double length = func(angleFrom * M_PI / 180) * segmentWidth;
    path.moveTo(xCenter + length * cos(angleFrom * M_PI / 180), yCenter - length * sin(angleFrom * M_PI / 180));

    for(double angle = angleFrom; angle <= angleTo; angle += step) {
        double length = func(angle * M_PI / 180) * segmentWidth;
        path.lineTo(xCenter + length * cos(angle * M_PI / 180), yCenter - length * sin(angle * M_PI / 180));
    }

    p.drawPath(path);
}

void MyGraph::paintEvent(QPaintEvent*)
{
    QPainter p(this);

    QFont textFont = p.font();
    textFont.setPointSize(fontSize);
    p.setFont(textFont);

    p.setRenderHint(QPainter::Antialiasing, true);

    p.setPen(QPen(Qt::black,3,Qt::SolidLine));
    drawAxis(p);

    p.setPen(QPen(Qt::red, 2, Qt::SolidLine));
    drawFunc(p);
}
