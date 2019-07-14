#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QWidget>

#include "mycone.h"

class MyWidget : public QWidget
{
private:
    bool isometric;
    MyCone cone;

protected:
    void keyPressEvent(QKeyEvent *event);
    void paintEvent(QPaintEvent *event);

public:
    explicit MyWidget(int count, double upRad, double lowRad, double height);

    void approximateBy(int _count);
};

#endif // MYWIDGET_H
