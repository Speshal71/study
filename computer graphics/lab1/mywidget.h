#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QtWidgets>
#include "mygraph.h"


class MyWidget: public QWidget
{
private:
    QWidget mainWindow;
    QVBoxLayout mainLayout;

    QHBoxLayout spinBoxesLayout;
    QSpinBox paramBox;
    QSpinBox angleFromBox;
    QSpinBox angleToBox;

    MyGraph graph;

public:
    MyWidget()
    {
        paramBox.setPrefix("parameter = ");
        paramBox.setRange(1, 10000);
        paramBox.setSingleStep(1);
        paramBox.setValue(1);
        paramBox.setFixedWidth(200);

        angleFromBox.setPrefix("from = ");
        angleFromBox.setRange(1, 10000);
        angleFromBox.setSingleStep(1);
        angleFromBox.setValue(1);
        angleFromBox.setFixedWidth(200);

        angleToBox.setPrefix("to = ");
        angleToBox.setRange(1, 10000);
        angleToBox.setSingleStep(1);
        angleToBox.setValue(360);
        angleToBox.setFixedWidth(200);

        QObject::connect(&paramBox, QOverload<int>::of(&QSpinBox::valueChanged) , [&graph](int i) {graph.changeParameter(i);});
        QObject::connect(&angleToBox, QOverload<int>::of(&QSpinBox::valueChanged) , [&graph](int i) {graph.changeAngleTo(i);});
        QObject::connect(&angleFromBox, QOverload<int>::of(&QSpinBox::valueChanged) , [&graph](int i) {graph.changeAngleFrom(i);});

        spinBoxesLayout.addWidget(&paramBox);
        spinBoxesLayout.addWidget(&angleFromBox);
        spinBoxesLayout.addWidget(&angleToBox);

        mainLayout.addLayout(&spinBoxesLayout);
        mainLayout.addWidget(&graph);

        mainWindow.setLayout(&mainLayout);
    }
};

#endif // MYWIDGET_H
