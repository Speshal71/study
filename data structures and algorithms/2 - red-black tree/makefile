prog: main.o ttree.o
	g++ -std=c++11 -o prog main.o ttree.o

main.o: main.cpp ttree.h
	g++ -std=c++11 -c main.cpp

ttree.o: ttree.cpp ttree.h
	g++ -std=c++11 -c ttree.cpp