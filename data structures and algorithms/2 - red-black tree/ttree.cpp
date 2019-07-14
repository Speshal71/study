#include "ttree.h"

#define BLACK true
#define RED false
#define EQUIV(x) (x == 0)
#define LESS(x) (x < 0)
#define GREAT(x) (x > 0)

Node* TTree::Barier;

TTree::TTree()
{
    Barier = new Node;
    Barier->Color = BLACK;
    Barier->Word = nullptr;
    Barier->WordSize = 0;
    Barier->Val = 0;
    Barier->Parent = nullptr;
    Barier->Left = nullptr;
    Barier->Right = nullptr;
    Head = Barier;
}

Node* TTree::Min(Node *n)
{
    while(n->Left != Barier) {
        n = n->Left;
    }
    return n;
}

Node TTree::Min()
{
    return *(Min(Head));
}

Node* TTree::Max(Node *n)
{
    while(n->Right != Barier) {
        n = n->Right;
    }
    return n;
}

Node TTree::Max()
{
    return *(Max(Head));
}

Node* TTree::Successor(Node *n)
{
    if (n->Right != Barier) {
        return Min(n->Right);
    }
    Node *parent = n->Parent;
    while(parent != Barier && n == parent->Right) {
        n = parent;
        parent = parent->Parent;
    }
    return parent;
}

void TTree::LeftRotate(Node *n)
{
    Node *nn = n->Right;
    n->Right = nn->Left;
    if (nn->Left != Barier) {
        nn->Left->Parent = n;
    }
    nn->Parent = n->Parent;
    if(n->Parent == Barier) {
        Head = nn;
    } else if (n == n->Parent->Right) {
        n->Parent->Right = nn;
    } else {
        n->Parent->Left = nn;
    }
    nn->Left = n;
    n->Parent = nn;
}

void TTree::RightRotate(Node *n)
{
    Node *nn = n->Left;
    n->Left = nn->Right;
    if (nn->Right != Barier) {
        nn->Right->Parent = n;
    }
    nn->Parent = n->Parent;
    if(n->Parent == Barier) {
        Head = nn;
    } else if (n == n->Parent->Right) {
        n->Parent->Right = nn;
    } else {
        n->Parent->Left = nn;
    }
    nn->Right = n;
    n->Parent = nn;
}

void TTree::FixUpAdd(Node *n)
{
    Node *uncle;

    while(n->Parent->Color == RED) {
        if(n->Parent == n->Parent->Parent->Left) {
            uncle = n->Parent->Parent->Right;
            if (uncle->Color == RED) {
                uncle->Color = BLACK;
                n->Parent->Color = BLACK;
                n = n->Parent->Parent;
                n->Color = RED;
            } else if(n == n->Parent->Right) {
                n = n->Parent;
                LeftRotate(n);
            } else {
                n->Parent->Color = BLACK;
                n->Parent->Parent->Color = RED;
                RightRotate(n->Parent->Parent);
            }
        } else {
            uncle = n->Parent->Parent->Left;
            if (uncle->Color == RED) {
                uncle->Color = BLACK;
                n->Parent->Color = BLACK;
                n = n->Parent->Parent;
                n->Color = RED;
            } else if(n == n->Parent->Left) {
                n = n->Parent;
                RightRotate(n);
            } else {
                n->Parent->Color = BLACK;
                n->Parent->Parent->Color = RED;
                LeftRotate(n->Parent->Parent);
            }
        }
    }

    Head->Color = BLACK;
}

bool TTree::Add(char *word, int wordSize, unsigned long int val)
{
    if (Head == Barier) {
        Head = new Node;
        Head->Color = BLACK;
        Head->Word = word;
        Head->WordSize = wordSize;
        Head->Val = val;
        Head->Parent = Barier;
        Head->Left = Barier;
        Head->Right = Barier;
    } else {
        Node *nn = Head;
        Node *n;
        int comp;

        while(nn != Barier) {
            n = nn;
            comp = strcmp(word, nn->Word);
            if (EQUIV(comp)) {
                free(word);
                return false;
            } else if(LESS(comp)) {
                nn = nn->Left;
            } else {
                nn = nn->Right;
            }
        }

        if(LESS(comp)) {
            n->Left = new Node;
            nn = n->Left;
        } else {
            n->Right = new Node;
            nn = n->Right;
        }

        nn->Color = RED;
        nn->Word = word;
        nn->WordSize = wordSize;
        nn->Val = val;
        nn->Parent = n;
        nn->Left = Barier;
        nn->Right = Barier;

        FixUpAdd(nn);
    }
    return true;
}

std::pair<bool, unsigned long int> TTree::Search(char *word)
{
    Node *nn = Head;
    int comp;
    while(nn != Barier) {
        comp = strcmp(word, nn->Word);
        if (EQUIV(comp)) {
            return std::pair<bool, unsigned long int>(true, nn->Val);
        } else if(LESS(comp)) {
            nn = nn->Left;
        } else {
            nn = nn->Right;
        }
    }
    return std::pair<bool, unsigned long int>(false, 0);
}

void TTree::FixUpDelete(Node *n)
{
    while(n != Head && n->Color == BLACK) {
        if (n == n->Parent->Left) {
            Node *brother = n->Parent->Right;
            if (brother->Color == RED) {
                brother->Color = BLACK;
                n->Parent->Color = RED;
                LeftRotate(n->Parent);
                brother = n->Parent->Right;
            }
            if (brother->Left->Color == BLACK && brother->Right->Color == BLACK) {
                brother->Color = RED;
                n = n->Parent;
            } else {
                if (brother->Right->Color == BLACK) {
                    brother->Left->Color = BLACK;
                    brother->Color = RED;
                    RightRotate(brother);
                    brother = n->Parent->Right;
                }
                brother->Color = n->Parent->Color;
                n->Parent->Color = BLACK;
                brother->Right->Color = BLACK;
                LeftRotate(n->Parent);
                n = Head;
            }
        } else {
            Node *brother = n->Parent->Left;
            if (brother->Color == RED) {
                brother->Color = BLACK;
                n->Parent->Color = RED;
                RightRotate(n->Parent);
                brother = n->Parent->Left;
            }
            if (brother->Left->Color == BLACK && brother->Right->Color == BLACK) {
                brother->Color = RED;
                n = n->Parent;
            } else {
                if (brother->Left->Color == BLACK) {
                    brother->Right->Color = BLACK;
                    brother->Color = RED;
                    LeftRotate(brother);
                    brother = n->Parent->Left;
                }
                brother->Color = n->Parent->Color;
                n->Parent->Color = BLACK;
                brother->Left->Color = BLACK;
                RightRotate(n->Parent);
                n = Head;
            }
        }
    }
    n->Color = BLACK;
}

bool TTree::Delete(char *word)
{
    Node *deleted = Head;
    int comp;

    while(deleted != Barier) {
        comp = strcmp(word, deleted->Word);
        if (EQUIV(comp)) {
            Node *replaced;
            Node *son;
            if (deleted->Left == Barier || deleted->Right == Barier) {
                replaced = deleted;
            } else {
                replaced = Successor(deleted);
            }
            if (replaced->Left != Barier) {
                son = replaced->Left;
            } else {
                son = replaced->Right;
            }
            son->Parent = replaced->Parent;
            if (replaced->Parent == Barier) {
                Head = son;
            } else if (replaced == replaced->Parent->Right) {
                replaced->Parent->Right = son;
            } else {
                replaced->Parent->Left = son;
            }
            if (replaced != deleted) {
                free(deleted->Word);
                deleted->Val = replaced->Val;
                deleted->Word = replaced->Word;
                deleted->WordSize = replaced->WordSize;
                replaced->Word = nullptr;
            }

            if (replaced->Color == BLACK) {
                FixUpDelete(son);
            }

            free(replaced->Word);
            delete replaced;

            return true;
        } else if(LESS(comp)) {
            deleted = deleted->Left;
        } else {
            deleted = deleted->Right;
        }
    }
    return false;
}

void TTree::Print(std::ostream &out, const Node *n, int depth = 0)
{
    if(n == Barier) {
        return;
    }

    Print(out, n->Right, depth + 4);

    for (int i = 0; i < depth; ++i) {
        out << " ";
    }
    out << n->Word << " " << n->Val;
    out << (n->Color == RED) ? 'r' : 'b' << '\n';

    Print(out, n->Left, depth + 4);
}

void TTree::Print(std::ostream &out)
{
    Print(out, Head);
}

void TTree::DeleteTree(Node *n)
{
    if (n == Barier) {
        return;
    }
    DeleteTree(n->Left);
    DeleteTree(n->Right);
    free(n->Word);
    delete n;
}

TTree::~TTree()
{
    DeleteTree(Head);
    delete Barier;
}

void TTree::SaveNode(std::ofstream &fout, Node *n)
{
    if (n == Barier) {
        return;
    }
    if (n->Left != Barier) {
        fout.write((char*)&(n->Left->Val), sizeof(n->Left->Val));
        fout.write((char*)&(n->Left->WordSize), sizeof(int));
        fout.write((char*)(n->Left->Word), n->Left->WordSize * sizeof(char));
    }
    if (n->Right != Barier) {
        fout.write((char*)&(n->Right->Val), sizeof(n->Right->Val));
        fout.write((char*)&(n->Right->WordSize), sizeof(int));
        fout.write((char*)(n->Right->Word), n->Right->WordSize * sizeof(char));
    }
    SaveNode(fout, n->Left);
    SaveNode(fout, n->Right);
}

bool TTree::Save(const char *ch)
{
    std::ofstream fout(ch, std::ios_base::binary);
    if (!fout.is_open()) {
        return false;
    }
    if (Head != Barier) {
        fout.write((char*)&(Head->Val), sizeof(Head->Val));
        fout.write((char*)&(Head->WordSize), sizeof(int));
        fout.write((char*)(Head->Word), Head->WordSize * sizeof(char));
        SaveNode(fout, Head);
    }
    fout.close();
    return true;
}

bool TTree::Load(const char *ch)
{
    unsigned long int val;
    char *word;
    int wordSize;

    std::ifstream fin(ch, std::ios_base::binary);
    if (!fin.is_open()) {
        return false;
    }
    DeleteTree(Head);
    Head = Barier;
    while(true) {
        fin.read((char*)&(val), sizeof(sizeof(val)));
        if (!fin.eof()) {
            fin.read((char*)&(wordSize), sizeof(int));
            word = (char *) malloc(wordSize * sizeof(char));
            fin.read((char*)(word), wordSize * sizeof(char));
            Add(word, wordSize, val);
        } else {
            break;
        }
    }
    fin.close();
    return true;
}