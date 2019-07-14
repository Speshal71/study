#include <QtWidgets>
#include "mygraph.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QWidget mainWindow;
    QVBoxLayout mainLayout;

    QHBoxLayout spinBoxesLayout;
    QSpinBox paramBox;
    QSpinBox angleFromBox;
    QSpinBox angleToBox;

    MyGraph graph;

    paramBox.setPrefix("parameter = ");
    paramBox.setRange(1, 10000);
    paramBox.setSingleStep(1);
    paramBox.setValue(1);

    angleFromBox.setPrefix("from = ");
    angleFromBox.setRange(1, 10000);
    angleFromBox.setSingleStep(1);
    angleFromBox.setValue(1);

    angleToBox.setPrefix("to = ");
    angleToBox.setRange(1, 10000);
    angleToBox.setSingleStep(1);
    angleToBox.setValue(360);

    QObject::connect(&paramBox, QOverload<int>::of(&QSpinBox::valueChanged) , [&graph](int i) {graph.changeParameter(i);});
    QObject::connect(&angleFromBox, QOverload<int>::of(&QSpinBox::valueChanged) , [&graph](int i) {graph.changeAngleFrom(i);});
    QObject::connect(&angleToBox, QOverload<int>::of(&QSpinBox::valueChanged) , [&graph](int i) {graph.changeAngleTo(i);});

    spinBoxesLayout.addWidget(&paramBox);
    spinBoxesLayout.addWidget(&angleFromBox);
    spinBoxesLayout.addWidget(&angleToBox);

    mainLayout.addLayout(&spinBoxesLayout);
    mainLayout.addWidget(&graph);

    mainWindow.setLayout(&mainLayout);

    mainWindow.setWindowTitle("lab1");
    mainWindow.setGeometry(500, 400, 600, 600);

    mainWindow.show();

    return app.exec();
}
