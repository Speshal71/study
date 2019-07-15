#include "thuffmantree.h"

struct comparator
{
    bool operator() (const TNode *p1, const TNode *p2)
    {
        return p1->freq > p2->freq;
    }
};

THuffmanTree::THuffmanTree(std::map<alpChar, uint32_t> &statistics)
{
    BuildFromStat(statistics);
}

THuffmanTree::THuffmanTree()
{

}

const std::string& THuffmanTree::GetCode(alpChar key)
{
    return table[key];
}

alpChar THuffmanTree::GetChar(IBitStream &ibfs)
{
    TNode *node = root;
    char c;

    while(node->left != nullptr || node->right != nullptr) {
        ibfs >> c;
        node = (c == '1' ? node->right : node->left);
    }

    return node->item;
}

void THuffmanTree::BuildTable(TNode *node, std::string &path)
{
    if (node != nullptr) {
        if (node->right == nullptr && node->left == nullptr) {
            table[node->item] = path;
        }
        path.push_back('1');
        BuildTable(node->right, path);
        path.pop_back();

        path.push_back('0');
        BuildTable(node->left, path);
        path.pop_back();
    }
}

void THuffmanTree::BuildFromStat(std::map<alpChar, uint32_t> &statistics)
{
    if (root != nullptr) {
        delete root;
        table.clear();
    } else {
        uint16_t alpSize = statistics.size();
        std::priority_queue<TNode*, std::vector<TNode*>, comparator> pq;

        for(auto i: statistics) {
            pq.push(new TNode(i.first, i.second));
        }

        for(uint16_t i = 1; i < alpSize; ++i) {
            TNode *l = pq.top();
            pq.pop();
            TNode *r = pq.top();
            pq.pop();
            pq.push(new TNode(alpChar(), l->freq + r->freq, l, r));
        }

        root = pq.top();
        pq.pop();

        std::string path;
        BuildTable(root, path);
    }
}

void THuffmanTree::SaveTreeTo(OBitStream &obfs, TNode *node)
{
    if (node->right == nullptr && node->left == nullptr) {
        obfs << '1';
        obfs << (node->item.first == true ? '1' : '0');
        obfs.Write((char *) &(node->item.second), sizeof(char));
    } else {
        obfs << '0';
        SaveTreeTo(obfs, node->left);
        SaveTreeTo(obfs, node->right);
    }
}

void THuffmanTree::SaveTo(OBitStream &obfs)
{
    SaveTreeTo(obfs, root);
}

TNode * THuffmanTree::LoadTreeFrom(IBitStream &ibfs)
{
    char c;
    TNode *node;

    ibfs >> c;
    if (c == '1') {
        alpChar item;
        ibfs >> c;
        item.first = (c == '1' ? true : false);
        ibfs.Read((char *) &(item.second), sizeof(char));
        node = new TNode(item, 0, nullptr, nullptr);
    } else {
        node = new TNode(alpChar(), 0, nullptr, nullptr);
        node->left = LoadTreeFrom(ibfs);
        node->right = LoadTreeFrom(ibfs);
    }

    return node;
}

void THuffmanTree::LoadFrom(IBitStream &ibfs)
{
    root = LoadTreeFrom(ibfs);
}

THuffmanTree::~THuffmanTree()
{
    delete root;
}