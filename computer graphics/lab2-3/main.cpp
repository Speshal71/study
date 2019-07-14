#include <QtWidgets>

#include "lightconsts.h"
#include "mywidget.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QWidget mainWindow;
    QVBoxLayout mainLayout;

    MyWidget w(3, 25, 50, 80);
    w.setFocusPolicy(Qt::StrongFocus);

    QHBoxLayout controlLayout;
    QLabel approxLabel("Level of approximation: ");
    QSlider changeApproximation(Qt::Horizontal);

    changeApproximation.setFocusPolicy(Qt::NoFocus);
    changeApproximation.setTickInterval(50);
    changeApproximation.setSingleStep(1);

    QObject::connect(&changeApproximation, QOverload<int>::of(&QSlider::valueChanged) , [&w](int i) {w.approximateBy(2 + i);});

    QLabel powerLabel("power: ");
    QDoubleSpinBox powerBox;

    powerBox.setFocusPolicy(Qt::NoFocus);
    powerBox.setRange(0.0, 1.0);
    powerBox.setSingleStep(0.1);
    powerBox.setValue(1.0);

    QObject::connect(&powerBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&w](double i) {lightPower = i; w.update();});

    controlLayout.addWidget(&powerLabel);
    controlLayout.addWidget(&powerBox);
    controlLayout.addWidget(&approxLabel);
    controlLayout.addWidget(&changeApproximation);

    mainLayout.addLayout(&controlLayout);
    mainLayout.addWidget(&w);

    mainWindow.setLayout(&mainLayout);

    mainWindow.setWindowTitle("lab3 Skvortsov");
    mainWindow.setGeometry(500, 400, 600, 600);

    mainWindow.show();

    w.show();

    return a.exec();
}
