#include "tbitstream.h"

OBitStream::OBitStream(std::ofstream &_ofs): ofs(_ofs), buff(0), size(0) {}
    
OBitStream& OBitStream::operator<<(char c)
{
    if (size == capacity) {
        Flush();
    }
    buff = (buff << 1) | (c - '0');
    ++size;

    return *this;
}

OBitStream& OBitStream::operator<<(const std::string &str)
{
    for (uint16_t i = 0; i < str.size(); ++i) {
        if (size == capacity) {
            Flush();
        }
        buff = (buff << 1) | (str[i] - '0');
        ++size;
    }

    return *this;
}

void OBitStream::Write(const char *str, uint16_t length)
{
    for (uint16_t i = 0; i < length; ++i) {
        char c = str[i]; 
        uint8_t temp_size = size;
        buff = (buff << (capacity - size)) | (((unsigned char) c) >> size);
        size = capacity;
        Flush();
        buff = c & (((unsigned char) 255) >> (capacity - temp_size));
        size = temp_size;
    }
}

bool OBitStream::Eof()
{
    return ofs.eof();
}

void OBitStream::Flush()
{
    if (size != 0) {
         buff = buff << (capacity - size);
        ofs.write((char *) &buff, sizeof(char));
        buff = 0;
        size = 0;
    }
}

void OBitStream::Close() 
{
    Flush();
}


OBitStream::~OBitStream() 
{
    Flush();
}

//----------------------------------------------------------

IBitStream::IBitStream(std::ifstream &_ifs): ifs(_ifs), buff(0), size(0) {}

IBitStream& IBitStream::operator>>(char &c)
{
    if (size == 0) {
        ifs.read((char *) &buff, sizeof(char));
        size = capacity;
    }
    c = (buff & 128) == 128 ? '1' : '0';
    buff <<= 1;
    --size;

    return *this;
}

void IBitStream::Read(char *str, uint16_t length)
{
    for (uint16_t i = 0; i < length; ++i) {
        str[i] = buff;
        ifs.read((char *) &buff, sizeof(char));
        str[i] = str[i] | (((unsigned char) buff) >> size);
        buff = buff << (capacity - size);
    }
}

bool IBitStream::Eof()
{
    return ifs.eof();
}

IBitStream::~IBitStream() {}