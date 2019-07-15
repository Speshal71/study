#include "tsufftree.h"

TSuffTree::TSuffTree(int _capacity) :
    root(new TSTNode()), longestSuffix(nullptr), lastLeaf(nullptr), pos(root), depth(1), 
    capacity(_capacity), size(0), str(2 * _capacity), updateCount(0)
{

}

bool TSuffTree::Go(char c)
{
    if (depth < pos->len && depth < str.DistanceToEnd(pos->beg)) { //position on the edge
        if (str[pos->beg + depth] == c) {
            ++depth;
            return true;
        }
    } else { //position on the node
        if (pos->edges.count(c)) {
            pos = pos->edges[c];
            depth = 1;
            return true;
        }
    }

    return false;
}

void TSuffTree::FastGo(int beg, int len)
{
    while (len > pos->len) {
        len -= pos->len;
        beg += pos->len;
        pos = pos->edges[str[beg]];
    }

    depth = len;
}

void TSuffTree::Split()
{
    if (depth < pos->len) { //position on the edge
        TSTNode *mid = new TSTNode(pos->parent,  pos->beg, depth);
        pos->parent->edges[str[pos->beg]] = mid;
        mid->edges[str[mid->beg + mid->len]] = pos;
        pos->parent = mid;
        pos->beg = mid->beg + mid->len;
        pos->len = pos->len - mid->len;

        pos = mid;
    }
}

TSTNode* TSuffTree::Fork(char c)
{
    Split();
    
    return (pos->edges[c] = new TSTNode(pos, str.EndPos(), capacity));
}

void TSuffTree::FindSuffRefPos()
{
    int shift = (pos->parent == root ? 1 : 0);
    int fastGoBeg = pos->beg + shift;
    int fastGoLen = depth - shift;

    if (fastGoLen == 0) {
        pos = root;
    } else {
        pos = pos->parent->suffRef->edges[str[fastGoBeg]];
        FastGo(fastGoBeg, fastGoLen);
    }
}

void TSuffTree::GoSuffRef()
{
    if (pos->suffRef == nullptr) {
        TSTNode *from = pos;

        FindSuffRefPos();
        Split();

        from->suffRef = pos;
    } else {
        pos = pos->suffRef;
    }
}

void TSuffTree::RemoveLeaf(TSTNode *n)
{
    if (n == nullptr || n->edges.size() > 0) {
        return;
    }

    char c = str[n->beg]; //c is the first letter leading to leaf
    n = n->parent;        //now n is internal node leading to leaf
    delete n->edges[c];
    n->edges.erase(c);

    if (n->edges.size() == 1 && n != root) { //remove internal node with the only child
        TSTNode *onlyLeaf = n->edges.begin()->second;

        n->parent->edges[str[n->beg]] = onlyLeaf;
        onlyLeaf->parent = n->parent;
        onlyLeaf->beg -= n->len;
        onlyLeaf->len += n->len;

        if (pos == onlyLeaf) {
            depth += n->len;
        }

        if (pos == n) {
            pos = onlyLeaf;
        }

        n->edges.begin()->second = nullptr;
        delete n;
    }
}

void TSuffTree::RemoveLongestSuffix()
{    
    if (longestSuffix != nullptr) {
        if (pos != longestSuffix) {
            TSTNode *n = longestSuffix;
            longestSuffix = longestSuffix->suffRef;
            RemoveLeaf(n);
        } else { //builder pos is on the longest suffix, special case
            //repoint longest suffix pos
            lastLeaf->suffRef = pos;
            lastLeaf = pos;
            longestSuffix = longestSuffix->suffRef;

            //relabel current node as a new one and go to the next one
            pos->beg = str.EndPos() + 1 - depth;
            pos->len = capacity;
            
            FindSuffRefPos(); //go to new pos
        }
    }
}

int TSuffTree::UpdateLabel(TSTNode *n)
{
    if (n->edges.empty()) {
        return str.DistanceToEnd(n->beg);
    }

    int minDist = capacity;
    int newEnd = n->beg;

    for (auto i: n->edges) {
        int dist = UpdateLabel(i.second);
        if (dist < minDist) {
            minDist = dist;
            newEnd = i.second->beg;
        }
    }

    n->beg = str.RealIndex(newEnd - n->len);
    return minDist + n->len;
}

void TSuffTree::UpdateLabels()
{
    for (auto i: root->edges) {
        UpdateLabel(i.second);
    }
}

void TSuffTree::Extend(char c)
{
    if (size == capacity) {
        RemoveLongestSuffix();
        --size;
    }
    str.Push(c);
    ++size;
    ++updateCount;

    if (size > 1) {
        while (!Go(c)) { //while rule 2
            TSTNode *newLeaf = Fork(c);

            //save pointers to the next longest suffix after the previous one
            lastLeaf->suffRef = newLeaf;
            lastLeaf = newLeaf;
            
            if (pos == root) {
                break;
            }
            GoSuffRef();
        }
    } else {
        longestSuffix = Fork(c);
        lastLeaf = longestSuffix;
    }

    if (updateCount == capacity) {
        UpdateLabels();
        updateCount;
        updateCount = 0;
    }
}

std::pair<int, int> TSuffTree::Find(TRingArray<char> &pattern)
{    
    // we are saving the builder position
    // because we will use tree's position pointers for searching
    TSTNode *savedPos = pos; 
    int savedDepth = depth;

    pos = root;
    depth = 1;

    int len;
    for (len = 0; len < pattern.size && Go(pattern[pattern.pos + len]); ++len) {}

    std::pair<int, int> ret(str.DistanceToEnd(pos->beg + depth - len), len);
    
    pos = savedPos;
    depth = savedDepth;

    return ret;
}

TSuffTree::~TSuffTree()
{
    delete root;
}