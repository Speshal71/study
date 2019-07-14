#include "mywidget.h"

#include <QDebug>

MyWidget::MyWidget()
{
    points.push_back(QRect(width() * 1 / 6, height() / 2 - height() / 3, 10, 10));
    points.push_back(QRect(width() * 2 / 6, height() / 2 + height() / 4, 10, 10));
    points.push_back(QRect(width() * 3 / 6, height() / 2, 10, 10));
    points.push_back(QRect(width() * 4 / 6, height() / 2 - height() / 5, 10, 10));
    points.push_back(QRect(width() * 5 / 6, height() / 2 + height() / 4, 10, 10));
}

QPoint MyWidget::BSpline(double u)
{
    double x, y;
    if (u >= 0 && u < 2) {
        x = pow(2 - u, 3) / 8.0 * points[0].center().x()
          + (19 * u * u * u - 90 * u * u + 108 * u) / 72.0 * points[1].center().x()
          + (18 * u * u - 7 * u * u * u) / 36.0 * points[2].center().x()
          + u * u * u / 18 * points[3].center().x();
        y = pow(2 - u, 3) / 8.0 * points[0].center().y()
          + (19 * u * u * u - 90 * u * u + 108 * u) / 72.0 * points[1].center().y()
          + (18 * u * u - 7 * u * u * u) / 36.0 * points[2].center().y()
          + u * u * u / 18 * points[3].center().y();
    } else if (u >= 2 && u < 3) {
        x = pow(3 - u, 3) / 9.0 * points[1].center().x()
          + (5 * u * u * u - 36 * u * u + 81 * u - 54) / 9.0 * points[2].center().x()
          + (-13 * u * u * u + 81 * u * u - 162 * u + 108) / 9.0 * points[3].center().x()
          + pow(u - 2, 3) * points[4].center().x();
        y = pow(3 - u, 3) / 9.0 * points[1].center().y()
          + (5 * u * u * u - 36 * u * u + 81 * u - 54) / 9.0 * points[2].center().y()
          + (-13 * u * u * u + 81 * u * u - 162 * u + 108) / 9.0 * points[3].center().y()
          + pow(u - 2, 3) * points[4].center().y();
    } else {
        x = 0;
        y = 0;
    }

    return QPoint(x, y);
}

void MyWidget::paintEvent(QPaintEvent* event)
{
    QPainter p(this);

    QPainterPath path;
    path.moveTo(BSpline(0));
    for(double u = 0.1; u <= 3.0; u += 0.05) {
        path.lineTo(BSpline(u));
    }
    p.setPen(QPen(Qt::green, 2));
    p.drawPath(path);

    for (int i = 1; i < points.size(); ++i) {
        p.setPen(QPen(Qt::gray));
        p.drawLine(points[i - 1].center(), points[i].center());
    }

    for (int i = 0; i < points.size(); ++i) {
        p.fillRect(points[i], Qt::red);
    }

}

void MyWidget::mousePressEvent(QMouseEvent* event)
{
    for (int i = 0; i < points.size(); ++i) {
        if (points[i].contains(event->pos())) {
            moveablePoint = i;
            isMoving = true;
            break;
        }
    }
}

void MyWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (isMoving) {
        points[moveablePoint].moveCenter(event->pos());

        update();
    }
}

void MyWidget::mouseReleaseEvent(QMouseEvent* event)
{
    isMoving = false;
}

