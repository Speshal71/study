#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#include "ttree.h"

char* getWord(int *wordSize)
{
    char c;
    char *str;
    int size = 0;
    str = (char*) malloc(260 * sizeof(char));
    while((c = getchar()) == ' ' || c == '\n') {}
    str[size++] = tolower(c);
    while((str[size] = tolower(getchar())) != ' ' && str[size] != '\n') {
        size++;
    }
    size++;
    str = (char*) realloc(str, size * sizeof(char));
    str[size - 1] = '\0';
    *wordSize = size; 
    return str;
}

void formattedStr(char *str)
{
    int i = 0;
    while((str[i] = tolower(str[i])) != '\0') {
        i++;
    }
}

int main()
{
    TTree tree;
    unsigned long int val;
    char word[260];
    char *str;
    int wordSize;

    while(std::cin >> word) {
        if (word[0] != '\n') {
            switch(word[0]) {
                case '+':
                    str = getWord(&wordSize);
                    (std::cin >> val).ignore();
                    if (tree.Add(str, wordSize, val)) {
                        std::cout << "OK\n";
                    } else {
                        std::cout << "Exist\n";
                    }
                    break;
                case '-':
                    std::cin >> word;
                    formattedStr(word);
                    if (tree.Delete(word)) {
                        std::cout << "OK\n";
                    } else {
                        std::cout << "NoSuchWord\n";
                    }
                    break;
                case '?':
                    tree.Print(std::cout);
                    break;
                case '!':
                    std::cin >> word;
                    if (strcmp(word, "Load") == 0) {
                        std::cin >> word;
                        if (tree.Load(word)) {
                            std::cout << "OK\n";
                        } else {
                            std::cout << "ERROR: Couldn't load file\n";
                        }
                    } else {
                        std::cin >> word;
                        if (tree.Save(word)) {
                            std::cout << "OK\n";
                        } else {
                            std::cout << "ERROR: Couldn't create file\n";
                        }
                    }
                    break;
                default:
                    formattedStr(word);
                    std::pair<bool, unsigned long int> res = tree.Search(word);
                    if (res.first) {
                        std::cout << "OK: " << res.second << '\n';
                    } else {
                        std::cout << "NoSuchWord\n";
                    }
                    
                    break;
            }
        }
    }

    return 0;
}