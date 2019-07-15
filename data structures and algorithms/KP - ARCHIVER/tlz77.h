#ifndef LZ77_H
#define LZ77_H

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "tbitstream.h"
#include "tringarray.h"
#include "tsufftree.h"

typedef uint8_t len_t;
typedef uint16_t dis_t;

const len_t MAX_INPUT_CAPACITY = 255;
const dis_t MAX_SEARCH_CAPACITY = 16384 / 2;
const char TERM = '\0';
const std::pair<bool, char> TERM_PAIR = {false, 0};

class TLZ77
{
private:
    len_t input_cap;
    dis_t search_cap;

public:
    TLZ77(len_t icap = MAX_INPUT_CAPACITY, dis_t scap = MAX_SEARCH_CAPACITY);

    std::map<std::pair<bool, char>, uint32_t> Encode(std::ifstream &from, std::ofstream &to);
    void Decode(std::ifstream &from, std::ofstream &to);

    ~TLZ77();
};

#endif