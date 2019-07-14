#include <iostream>
#include <vector>
#include <cstring>

const int alpSize = 100;

std::vector<int> BuildBCTable(std::vector<unsigned int> &s)
{
    std::vector<int> bmBC(s.size(), 0);

    for(int i = 0; i < (s.size() - 1); i++) {
        bmBC[s[i] % alpSize] = i;
    }

    return bmBC;
}

std::vector<int> BuildSLTable(std::vector<unsigned int> &s)
{
    int veclen = s.size();
    std::vector<int> bmSL(s.size(), 0);
    int l = veclen - 1;
    int r = veclen - 1;

    for (int i = veclen - 2; i >= 0; --i) {
        if (i < l) {
            int k = i;
            int j = veclen - 1;
            while(k >= 0 && s[k] == s[j]) {
                k--;
                j--;
                bmSL[i]++;
            }
            if (bmSL[i] > 0) {
                l = i - bmSL[i] + 1;
                r = i;
            } 
        } else {
            int ii = veclen - 1 - r + i;
            if (bmSL[ii] < (i - l + 1)) {
                bmSL[i] = bmSL[ii];
            } else {
                int k = l - 1;
                int j = veclen - 1 - bmSL[ii];
                bmSL[i] = i - l + 1;
                while(k >= 0 && s[k] == s[j]) {
                    k--;
                    j--;
                    bmSL[i]++;
                }
                if (bmSL[i] > 0) {
                    l = i - bmSL[i] + 1;
                    r = i;
                }
            }
        }
    }

    return bmSL;
}

std::vector<int> BuildGSTable(std::vector<int> &bmSL)
{
    int veclen = bmSL.size();
    std::vector<int> bmGS(veclen, 0);

    for (int j = 0; j < (veclen - 1); ++j) {
        int i = veclen - bmSL[j];
        if (i != veclen)
            bmGS[i] = j;
    }

    return bmGS;
}

std::vector<int> BuildPSTable(std::vector<int> &bmSL)
{
    int veclen = bmSL.size();
    std::vector<int> bmPS(veclen, 0);

    for (int i = veclen - 1; i >= 0; i--) {
        int j = veclen - i - 1;
        if (bmSL[j] == (j + 1))
            bmPS[i] = j + 1;
        else
            if (i != (veclen - 1))
                bmPS[i] = bmPS[i + 1];
    }

    return bmPS;
}

void RealtimeSearchString(std::vector<unsigned int> &p)
{
    int plen = p.size();
    unsigned int *t = new unsigned int[plen];
    int *str  = new int[plen];
    int *pos  = new int[plen];
    int *m    = new int[plen];

    memset(m, 0, plen * sizeof(int));

    std::vector<int> bmBC = BuildBCTable(p);
    std::vector<int> bmSL = BuildSLTable(p);
    std::vector<int> bmGS = BuildGSTable(bmSL);
    std::vector<int> bmPS = BuildPSTable(bmSL);

    int strCount = 0;
    int strBegin = 0;
    int numCount = 0;
    unsigned int num;
    int shift = 0;
    char c;

    while (std::cin && ((c = std::cin.peek()) == '\n' || c == ' ' || c == '\t')) {
        std::cin.ignore();
        if (c == '\n')
            strCount++;
    }

    while(std::cin >> num) {
        numCount++;
        str[shift] = strCount;
        pos[shift] = numCount - strBegin;
        t[shift++] = num;

        //START SEARCHING

        if (shift == plen) {
            int i = plen - 1;
            while (i >= 0) {
                if (m[i] == 0) {
                    if (t[i] == p[i]) {
                        i--;
                    } else {
                        m[plen - 1] = plen - 1 - i;
                        break;
                    }
                } else if (m[i] < bmSL[i]) {
                    i -= m[i];
                } else if (m[i] >= bmSL[i] && bmSL[i] == (i + 1)) {
                    i = -1;
                    m[plen - 1] = plen - 1 - i;
                    break;
                } else if (m[i] > bmSL[i] && bmSL[i] <= i) {
                    m[plen - 1] = plen - 1 - i;
                    break; 
                } else if (m[i] == bmSL[i] && bmSL[i] < (i + 1)) {
                    i -= m[i];
                }
            }

            int offset;
            if (i < 0) {
                std::cout << str[0] << ", " << pos[0] << '\n';
                if (plen != 1)
                    offset = plen - bmPS[1];
                else
                    offset = 1;
            } else {
                if (i == (plen - 1))
                    offset = std::max(1, i - bmBC[t[i] % alpSize]);
                else
                    offset = std::max(std::max(1, i - bmBC[t[i] % alpSize]), plen - bmGS[i + 1] - 1);
            }

            memmove(m, m + offset, (plen - offset) * sizeof(int));
            memmove(t, t + offset, (plen - offset) * sizeof(unsigned int));
            memmove(str, str + offset, (plen - offset) * sizeof(unsigned int));
            memmove(pos, pos + offset, (plen - offset) * sizeof(unsigned int));
            
            memset(m + plen - offset, 0, offset * sizeof(int));


            shift = plen - offset; 
        }

        //END SEARCHING


        while (std::cin && ((c = std::cin.peek()) == '\n' || c == ' ' || c == '\t')) {
            std::cin.ignore();
            if (c == '\n') {
                strCount++;
                strBegin = numCount;
            }
        }
    }

    delete[] t;
    delete[] str;
    delete[] pos;
    delete[] m;
}

void EnterPattern(std::vector<unsigned int> &pattern)
{
    unsigned int num;
    char c;
    do {
        std::cin >> num;
        pattern.push_back(num);
        while((c = std::cin.peek()) == ' ' || c == '\t')
            std::cin.ignore();
    } while(std::cin.peek() != '\n');
}


int main()
{
    std::vector<unsigned int> pattern;
    
    EnterPattern(pattern);

    RealtimeSearchString(pattern);
}