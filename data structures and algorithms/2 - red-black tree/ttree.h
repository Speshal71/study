#ifndef __TTREE_H__
#define __TTREE_H__

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

typedef struct node {
    bool Color;
    char *Word;
    int WordSize;
    unsigned long int Val;
    struct node *Parent;
    struct node *Right;
    struct node *Left;
} Node;

class TTree
{
private:
    Node *Head;
    static Node *Barier;

    Node* Min(Node *n);
    Node* Max(Node *n);
    Node* Successor(Node *n);
    void LeftRotate(Node *n);
    void RightRotate(Node *n);
    void DeleteTree(Node *n);
    void FixUpAdd(Node *n);
    void FixUpDelete(Node *);
    void SaveNode(std::ofstream &fout, Node *n);
public:
    TTree();

    bool Add(char *word, int wordSize, unsigned long int val);
    bool Delete(char *word);
    void Print(std::ostream &out);
    void Print(std::ostream &out, const Node *n, int depth);
    std::pair<bool, unsigned long int> Search(char *word);
    bool Save(const char *ch);
    bool Load(const char *ch);
    Node Min();
    Node Max();

    ~TTree();
};

#endif