#include <iostream>
#include <vector>
#include <algorithm>

struct TSegment {
    int s, f, pos;
};

int main() {
    int n, f, s = 0, curSeg = -1;
    std::vector<TSegment> segments, covering;

    std::cin >> n;
    segments.resize(n);
    for(int i = 0; i < n; i++) {
        std::cin >> segments[i].s >> segments[i].f;
        segments[i].pos = i;
    }

    std::cin >> f;
    TSegment temp = {f + 1, f + 1, n};
    segments.push_back(temp);
    ++n;

    std::sort(segments.begin(), segments.end(), [](TSegment &s1, TSegment &s2) 
    {
        return s1.s < s2.s;
    });

    for (int i = 0; i < n && s < f; i++) {
        if (curSeg == -1) {
            if (segments[i].s <= s) {
                curSeg = i;
            } else {
                std::cout << "0\n";
                return 0;
            }
        } else {
            if (segments[i].s > s) {
                covering.push_back(segments[curSeg]);
                s = segments[curSeg].f;
                curSeg = -1;
                --i;
            } else if (segments[i].f > segments[curSeg].f) {
                curSeg = i;
            }
        }
    }

    std::sort(covering.begin(), covering.end(), [](TSegment &s1, TSegment &s2) 
    {
        return s1.pos < s2.pos;
    });

    std::cout << covering.size() << '\n';
    for (int i = 0; i < covering.size(); i++) {
        std::cout << covering[i].s << ' ' << covering[i].f << '\n';
    }

    return 0;
}