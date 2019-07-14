#include <iostream>
#include <vector>
#include <stack>

const int LEFT   = 0;
const int CENTER = 1;
const int RIGHT  = 2;

int main() {
    int n, m, start;
    std::vector<std::vector<int>> a;
    std::vector<std::vector<long long int>> b;
    std::vector<std::vector<char>> c;
    std::stack<int> path;

    std::cin >> n >> m;
    
    a.resize(n);
    b.resize(n);
    c.resize(n);
    for (int i = 0; i < n; i++) {
        a[i].resize(m, 0);
        b[i].resize(m, 0);
        c[i].resize(m, 0);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cin >> a[i][j];
        }
    }

    for(int i = 0; i < m; i++) {
        b[0][i] = a[0][i];
    }

    for(int i = 0; i < (n - 1); i++) {
        b[i + 1][0] = a[i + 1][0] + (b[i][0] <= b[i][1] ? b[i][0] : b[i][1]);
        c[i + 1][0] = (b[i][0] <= b[i][1] ? CENTER : RIGHT);

        for (int j = 1; j < (m - 1); j++) {
            char pos = (b[i][j - 1] <= b[i][j] ? LEFT : CENTER);
            long long int min = (b[i][j - 1] <= b[i][j] ? b[i][j - 1] : b[i][j]);

            pos = (min <= b[i][j + 1] ? pos : RIGHT);
            min = (min <= b[i][j + 1] ? min : b[i][j + 1]);

            b[i + 1][j] = a[i + 1][j] + min;
            c[i + 1][j] = pos;
        }

        b[i + 1][m - 1] = a[i + 1][m - 1] + (b[i][m - 2] <= b[i][m - 1] ? b[i][m - 2] : b[i][m - 1]);
        c[i + 1][m - 1] = (b[i][m - 2] <= b[i][m - 1] ? LEFT : CENTER);
    }

    start = 0;
    for (int i = 1; i < m; i++) {
        if (b[n - 1][i] < b[n - 1][start]) {
            start = i;
        }
    }

    std::cout << b[n - 1][start] << '\n';

    for (int i = n - 1; i >= 0; i--) {
        path.push(start);
        start = start + c[i][start] - 1;
    }

    for (int i = 0; i < (n - 1); i++) {
        std::cout << '(' << i + 1 << ',' << path.top() + 1 << ") ";
        path.pop();
    }

    std::cout << '(' << n << ',' << path.top() + 1 << ")\n";

    return 0;
}