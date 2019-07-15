#include "tlz77.h"

TLZ77::TLZ77(len_t icap, dis_t scap): input_cap(icap), search_cap(scap) {}

std::map<std::pair<bool, char>, uint32_t> TLZ77::Encode(std::ifstream &from, std::ofstream &to)
{
    OBitStream obfs(to);
    std::map<std::pair<bool, char>, uint32_t> statistics;
    TRingArray<char> input_buff(input_cap);
    TSuffTree search_buff(search_cap);
    char c;

    while(input_buff.size != input_cap && from.peek() != EOF) {
        from.read((char *) &c, sizeof(char));
        input_buff.Push(c);
    }

    while (input_buff.size != 0) {
        std::pair<dis_t, len_t> matching = search_buff.Find(input_buff);

        if (matching.second == 0) {
            matching.second = 1;
        }

        if (matching.second > 3) {
            obfs << "0";
            obfs.Write((char *) &matching.first, sizeof(dis_t));
            obfs.Write((char *) &matching.second, sizeof(len_t));
            ++statistics[std::pair<bool, char>(false, (char) matching.second)];
        } else {
            for (len_t i = 0; i < matching.second; ++i) {
                obfs << "1";
                c = input_buff[input_buff.pos + i];
                obfs.Write((char *) &c, sizeof(char));
                ++statistics[std::pair<bool, char>(true, c)];
            }
        }

        for (len_t i = 0; i < matching.second; ++i) {
            c = input_buff[input_buff.pos];
            search_buff.Extend(c);
            input_buff.PopFront();
        }

        while(input_buff.size != input_cap && from.peek() != EOF) {
            from.read((char *) &c, sizeof(char));
            input_buff.Push(c);
        }

    }

    statistics[TERM_PAIR] = 1;

    return statistics;
}

void TLZ77::Decode(std::ifstream &from, std::ofstream &to)
{
    IBitStream ifs(from);
    TRingArray<char> search_buff(MAX_SEARCH_CAPACITY * 2);
    char c;

    while (!ifs.Eof()) {
        ifs >> c;
        if (c == '0') {
            std::pair<dis_t, len_t> matching;

            ifs.Read((char *) &matching.first, sizeof(dis_t));
            ifs.Read((char *) &matching.second, sizeof(len_t));

            for (len_t i = 0; i < matching.second; ++i) {
                c = search_buff[search_buff.DistanceToEnd(0) - matching.first];
                to.write((char *) &c, sizeof(char));
                search_buff.Push(c);
            }
        } else {
            ifs.Read((char *) &c, sizeof(char));
            to.write((char *) &c, sizeof(char));
            search_buff.Push(c);
        }
    }
}

TLZ77::~TLZ77() {}