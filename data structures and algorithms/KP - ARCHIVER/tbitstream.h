#ifndef TBITSTREAM_H
#define TBITSTREAM_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

class OBitStream
{
private:
    std::ofstream &ofs;
    char buff;
    uint8_t size;
    const uint8_t capacity = 8;

public:
    OBitStream(std::ofstream &_ofs);
    
    OBitStream& operator<<(char c);
    OBitStream& operator<<(const std::string &str);
    void Write(const char *str, uint16_t size);
    bool Eof();
    void Flush();
    void Close();

    ~OBitStream();
};

class IBitStream
{
private:
    std::ifstream &ifs;
    char buff;
    uint8_t size;
    const uint16_t capacity = 8;

public:
    IBitStream(std::ifstream &_ifs);

    IBitStream& operator>>(char &c);
    void Read(char *str, uint16_t size);
    bool Eof();

    ~IBitStream();
};

#endif