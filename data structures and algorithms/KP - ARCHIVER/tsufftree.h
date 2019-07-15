#ifndef TSUFFTREE_H
#define TSUFFTREE_H

#include <iostream>
#include <map>
#include <vector>
#include <string>

#include "tringarray.h"

const int inf = 100000;

struct TSTNode
{
    TSTNode* parent;
    TSTNode* suffRef;
    std::map<char, TSTNode*> edges;
    int beg, len;

    TSTNode() :
        parent(this), suffRef(this), beg(0), len(0) {}

    TSTNode(TSTNode *_parent, int _beg, int _len) :
        parent(_parent), suffRef(nullptr), beg(_beg), len(_len) {}

    ~TSTNode()
    {
        for (auto i: edges) {
            delete i.second;
        }
    }
};

class TSuffTree
{
private:
    TSTNode *root;
    TSTNode *longestSuffix;
    TSTNode *lastLeaf;

    TRingArray<char> str;
    const int capacity;
    int size, updateCount;

    //pointers to current position in the tree
    TSTNode *pos; // <<<---
    int depth;  // <<<---

    //functions to control position movements and to modify the tree
    bool Go(char c);
    void FastGo(int beg, int len); //we use it when we sure the path exists
    void FindSuffRefPos();
    void GoSuffRef();

    //functions to modify the tree
    void Split();
    TSTNode* Fork(char c);
    void RemoveLeaf(TSTNode *n);
    void RemoveLongestSuffix();

    //update edges' labels to maintain their validaty
    void UpdateLabels();
    int UpdateLabel(TSTNode *n);
public:
    TSuffTree(int _capacity);

    void Extend(char c);
    std::pair<int, int> Find(TRingArray<char> &pattern);  //(indexes (i, len))

    ~TSuffTree();
};

#endif