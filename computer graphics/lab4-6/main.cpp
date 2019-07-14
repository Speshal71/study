#include <QtWidgets>

#include "mywidget.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QWidget mainWindow;
    QVBoxLayout mainLayout;

    MyWidget w(3, 0.25, 0.5, 1.0);
    w.setFocusPolicy(Qt::StrongFocus);

    QHBoxLayout controlLayout;

    QLabel approxLabel("Level of approximation: ");
    QSlider changeApproximation(Qt::Horizontal);

    changeApproximation.setFocusPolicy(Qt::NoFocus);
    changeApproximation.setTickInterval(50);
    changeApproximation.setSingleStep(1);

    QObject::connect(&changeApproximation, QOverload<int>::of(&QSlider::valueChanged) , [&w](int i) {w.approximateBy(3 + i);});

    QRadioButton animationButton("Play animation");

    animationButton.setFocusPolicy(Qt::NoFocus);

    QObject::connect(&animationButton, QOverload<bool>::of(&QRadioButton::toggled) , [&w]() {w.startAnimation();});

    controlLayout.addWidget(&approxLabel);
    controlLayout.addWidget(&changeApproximation);
    controlLayout.addWidget(&animationButton);

    mainLayout.addLayout(&controlLayout);
    mainLayout.addWidget(&w);

    mainWindow.setLayout(&mainLayout);

    mainWindow.setWindowTitle("lab4,5 Skvortsov");
    mainWindow.setGeometry(500, 400, 600, 600);

    mainWindow.show();

    return a.exec();
}
