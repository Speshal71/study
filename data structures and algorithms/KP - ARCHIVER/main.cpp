#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <map>
#include <experimental/filesystem>
#include <iomanip>

#include "tlz77.h"
#include "tbitstream.h"
#include "thuffmantree.h"

namespace fs = std::experimental::filesystem;

const char FAIL[] = "\0";
enum {COMPRESS = 'c', DECOMPRESS = 'd', KEEP = 'k'};
bool keep_flag = false;
char action_flag = COMPRESS;


void encode(std::ifstream &from, std::ofstream &to)
{
    //LZ77 Encoding
    std::ofstream temp_ofs("temp.bin", std::ios_base::binary);

    std::map<std::pair<bool, char>, uint32_t> statistics = TLZ77().Encode(from, temp_ofs);

    temp_ofs.close();


    //Huffman Encoding
    std::ifstream temp_ifs("temp.bin", std::ios_base::binary);
    IBitStream ibfs(temp_ifs);
    OBitStream obfs(to);
    char c;

    THuffmanTree ht(statistics);

    ht.SaveTo(obfs);

    while(!ibfs.Eof()) {
        ibfs >> c;
        if (c == '0') {
            std::pair<dis_t, len_t> matching;

            ibfs.Read((char *) &matching.first, sizeof(dis_t));
            ibfs.Read((char *) &matching.second, sizeof(len_t));

            obfs << ht.GetCode(std::pair<bool, char>(false, (char) matching.second));
            obfs.Write((char *) &matching.first, sizeof(dis_t));
        } else {
            ibfs.Read((char *) &c, sizeof(char));
            obfs << ht.GetCode(std::pair<bool, char>(true, c));
        }
    }
    obfs << ht.GetCode(TERM_PAIR);

    obfs.Close();
    temp_ifs.close();

    fs::remove(fs::path("temp.bin"));
}


void decode(std::ifstream &from, std::ofstream &to)
{
    //Huffman Decoding
    std::ofstream temp_ofs("temp.bin", std::ios_base::binary);
    IBitStream ibfs(from);
    OBitStream obfs(temp_ofs);

    THuffmanTree ht;
    std::pair<bool, char> temp;
    uintmax_t unzip_size;

    ht.LoadFrom(ibfs);

    while ((temp = ht.GetChar(ibfs)) != TERM_PAIR) {
        if (temp.first == true) {
            obfs << '1';
            obfs.Write((char *) &temp.second, sizeof(char));
        } else {
            dis_t distance;
            ibfs.Read((char *) &distance, sizeof(dis_t));
            obfs << '0';
            obfs.Write((char *) &distance, sizeof(dis_t));
            obfs.Write((char *) &temp.second, sizeof(len_t));
        }
    }

    obfs.Close();
    temp_ofs.close();


    //LZ77 Decoding
    std::ifstream temp_ifs("temp.bin", std::ios_base::binary);
    
    TLZ77().Decode(temp_ifs, to);

    temp_ifs.close();

    fs::remove(fs::path("temp.bin"));
}


bool process(const std::string &filename)
{
    std::ifstream ifs(filename);

    if (!ifs.is_open()) {
        return false;
    }

    if (action_flag == COMPRESS) {
        std::ofstream ofs(filename + ".myzip");
        encode(ifs, ofs);
        ofs.close();
    }
    
    if (action_flag == DECOMPRESS) {
        std::string decompressed_name = filename.substr(0, filename.size() - 6);

        if (filename == decompressed_name + ".myzip") {
            std::ofstream ofs(decompressed_name);
            decode(ifs, ofs);
            ofs.close();
        } else {
            return false;
        }
    }
    
    ifs.close();

    if (keep_flag == false) {
        fs::remove(fs::path(filename));
    }

    return true;
}


bool parseFlags(char const *flags)
{
    if (flags[0] != '-' || flags[1] == '\0') {
        return false;
    }

    for(int i = 1; flags[i] != '\0'; ++i) {
        char c = flags[i];
        if (c == COMPRESS || c == DECOMPRESS) {
            action_flag = c;
        } else if (c == KEEP) {
            keep_flag = true;
        } else {
            return false;
        }
    }

    return true;
}


int main(int argc, char const *argv[])
{
    char c;
    if (argc == 1) {
        std::cout << "No input\n";
        std::cout << "For help, type: " << argv[0] << " -h\n";
        return 0;
    }

    if (parseFlags(argv[1]) == false) {
        std::cout << "Unknown flag\n";
        std::cout << "For help, type: " << argv[0] << " -h\n";
        return 0;
    }

    for (int i = 2; i < argc; ++i) {
        if (fs::exists(argv[i])) {
            if (fs::is_directory(fs::path(argv[i])))  {
                fs::recursive_directory_iterator begin(argv[i]);
                fs::recursive_directory_iterator end;

                for (auto path = begin; path != end; ++path) {
                    if (fs::is_regular_file(*path)) {
                        process(path->path().string());
                    }
                }
            } else {
                if (fs::is_regular_file(argv[i])) {
                    process(std::string(argv[i]));
                }
            }
        } else {
            std::cout << argv[0] << ": \"" << argv[i] << "\" -- No such file or directory\n";
        }
    }

    return 0;
}