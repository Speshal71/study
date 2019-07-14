#include <iostream>
#include "bigint.h"

int main()
{
    TBigInt i1, i2;
    char c;

    while(std::cin.peek() != EOF) {
        std::cin >> i1 >> i2;

        while((c = std::cin.get()) == ' ' || c == '\t' || c == '\n') {}

        switch(c) {
            case '+':
                std::cout << i1 + i2 << '\n';
                break;
            case '-':
                if (i1 < i2) {
                    std::cout << "Error\n";
                } else {
                    std::cout << i1 - i2 << '\n';
                }
                break;
            case '*':
                std::cout << i1 * i2 << '\n';
                break;
            case '/':
                if (i2 == 0) {
                    std::cout << "Error\n";
                } else {
                    std::cout << i1 / i2 << '\n';
                }
                break;
            case '^':
                if (i1 == 0 && i2 == 0) {
                    std::cout << "Error\n";
                } else {
                    std::cout << pow(i1, i2.ToInt()) << '\n';
                }
                break;
            case '>':
                std::cout << (i1 > i2 ? "true" : "false") << '\n';
                break;
            case '<':
                std::cout << (i1 < i2 ? "true" : "false") << '\n';
                break;
            case '=':
                std::cout << (i1 == i2 ? "true" : "false") << '\n';
                break;
        }
    }
}