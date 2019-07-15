myzip: main.o tbitstream.o thuffmantree.o tlz77.o tsufftree.o
	g++ -std=c++17 -O3 -o myzip main.o tbitstream.o thuffmantree.o tlz77.o tsufftree.o -lstdc++fs

main.o: main.cpp tringarray.h thuffmantree.h tbitstream.h tlz77.h
	g++ -std=c++17 -O3 -c main.cpp -lstdc++fs

tlz77.o: tlz77.cpp tlz77.h tbitstream.h tsufftree.h tringarray.h
	g++ -std=c++17 -O3 -c tlz77.cpp -lstdc++fs

thuffmantree.o: thuffmantree.cpp thuffmantree.h tbitstream.h
	g++ -std=c++17 -O3 -c thuffmantree.cpp -lstdc++fs

tbitstream.o: tbitstream.cpp tbitstream.h
	g++ -std=c++17 -O3 -c tbitstream.cpp -lstdc++fs

tsufftree.o: tsufftree.cpp tsufftree.h tringarray.h
	g++ -std=c++17 -O3 -c tsufftree.cpp -lstdc++fs