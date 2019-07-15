#ifndef THUFFMANTREE_H
#define THUFFMANTREE_H

#include <iostream>
#include <fstream>
#include <queue>
#include <vector>
#include <map>
#include <string>

#include "tbitstream.h"

typedef std::pair<bool, char> alpChar;

struct TNode
{
    alpChar item;
    uint32_t freq;
    TNode *left;
    TNode *right;

    TNode(alpChar _item, uint32_t _freq, TNode *_left = nullptr, TNode *_right = nullptr):
        item(_item), freq(_freq), left(_left), right(_right) {}

    ~TNode()
    {
        if (left != nullptr) {
            delete left;
        }
        if (right != nullptr) {
            delete right;
        }
    }
};

class THuffmanTree
{
private:
    TNode *root = nullptr;
    std::map<alpChar, std::string> table;

    void BuildTable(TNode *node, std::string &path);
    void SaveTreeTo(OBitStream &obfs, TNode *node);
    TNode * LoadTreeFrom(IBitStream &ibfs);

public:
    THuffmanTree(std::map<alpChar, uint32_t> &statistics);
    THuffmanTree();

    const std::string& GetCode(alpChar key);
    alpChar GetChar(IBitStream &ibfs);
    void BuildFromStat(std::map<alpChar, uint32_t> &statistics);
    void SaveTo(OBitStream &obfs);
    void LoadFrom(IBitStream &ibfs);

    ~THuffmanTree();
};

#endif 