#include "bigint.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <climits>

int& TBigInt::operator[](int k)
{
    return Num[k];
}

int TBigInt::operator[](int k) const
{
    return Num[k];
}

size_t TBigInt::size() const
{
    return Num.size();
}

int fdiv(int a, int b)
{
    return (a - (((a % b) + b) % b)) / b;
}

int fmod(int a, int b)
{
    return ((a % b) + b) % b;
}

TBigInt::TBigInt(int i)
{
    do {
        Num.push_back(i % BASE);
    } while((i /= BASE) != 0);
}

int TBigInt::ToInt()
{
    int ret = 0;

    if (*this > INT_MAX) {
        ret = INT_MAX;
    } else {
        int x = 1;
        for (int i = 0; i < this->size(); ++i) {
            ret = (*this)[i] * x + ret;
            x *= BASE;
        }
    }

    return ret;
}

std::istream& operator>>(std::istream& is, TBigInt& i)
{
    char c;

    i.Num.clear();

    while ((c = is.peek()) == '\n' || c == '\t' || c == ' ')
        is.ignore();
    
    while(is.peek() == '0')
        is.ignore();
    
    if (std::isdigit(is.peek())) {
        std::string str;

        is >> str;
        for (int k = str.length(); k > 0; k -= BASELEN) {
            if (k < BASELEN)
                i.Num.push_back(atoi(str.substr(0, k).data()));
            else
                i.Num.push_back(atoi(str.substr(k - BASELEN, BASELEN).data()));
        }
    } else {
        i.Num.push_back(0);
    }

    return is;
}

std::ostream& operator<<(std::ostream& os, const TBigInt& i)
{
    printf("%d", i[i.size() - 1]);

    for (int k = i.size() - 2; k >= 0; --k)
        printf("%0*d", BASELEN, i[k]);

    return os;
}

bool operator==(const TBigInt& i1, const TBigInt& i2)
{
    if (i1.size() == i2.size()) {
        for (int k = i1.size() - 1; k >= 0; --k)
            if (i1[k] != i2[k])
                return false;
    } else {
        return false;
    }
    
    return true;
}

bool operator>(const TBigInt& i1, const TBigInt& i2)
{
    if (i1.size() > i2.size()) {
        return true;
    } else if (i1.size() == i2.size()) {
        for (int k = i1.size() - 1; k >= 0; --k)
            if (i1[k] != i2[k])
                return i1[k] > i2[k] ? true : false;
    }

    return false;
}

bool operator<(const TBigInt& i1, const TBigInt& i2)
{
    return i2 > i1;
}

TBigInt& TBigInt::Plus(const TBigInt& i2, int l, int r)
{
    int carry = 0;

    for (int k = l; k <= r; k++) {
        int a = k < this->size() ? (*this)[k] : 0;
        int b = k < i2.size() ? i2[k - l] : 0;
        (*this)[k] = (a + b + carry) % BASE;
        carry = (a + b + carry) / BASE;
    }

    if (carry != 0) {
        if (r < (this->size() - 1))
            (*this)[l + 1] = carry;
        else
            this->Num.push_back(carry);
    }

    return *this;
}

TBigInt operator+(const TBigInt& i1, const TBigInt& i2)
{
    TBigInt ret = i1;
    int len = std::max(i1.size(), i2.size());
    
    ret.Num.resize(len, 0);
    ret.Plus(i2, 0, len - 1);

    return ret;
}

TBigInt& TBigInt::Minus(const TBigInt& i2, int l, int r)
{
    int carry = 0;

    for (int k = l; k <= r ; k++) {
        int a = k < this->size() ? (*this)[k] : 0;
        int b = (k - l) < i2.size() ? i2[k - l] : 0;
        (*this)[k] = fmod(a - b + carry, BASE);
        carry = fdiv(a - b + carry, BASE);
    }

    return *this;
}

TBigInt operator-(const TBigInt& i1, const TBigInt& i2)
{
    TBigInt ret = i1;
    int len = std::max(i1.size(), i2.size());

    ret.Minus(i2, 0, len - 1);

    while(ret.Num.back() == 0 && ret.size() > 1)
        ret.Num.pop_back();

    return ret;
}

TBigInt operator*(const TBigInt& i1, const TBigInt& i2)
{
    TBigInt ret;
    
    ret.Num.resize(i1.size() + i2.size(), 0);

    for (int k = 0; k < i1.size(); k++) {
        if (i1[k] != 0) {
            int carry = 0;
            for (int j = 0; j < i2.size(); j++) {
                int t = i1[k] * i2[j] + ret[k + j] + carry;
                ret[k + j] = t % BASE;
                carry = t / BASE;
            }
            ret[k + i2.size()] = carry;
        }
    }

    while(ret.Num.back() == 0 && ret.size() > 1)
        ret.Num.pop_back();

    return ret;
}

bool TBigInt::LessThen(const TBigInt& i2, int l, int r)
{
    for (int k = r; k >= l; k--) {
        int a = k < this->size() ? (*this)[k] : 0;
        int b = (k - l) < i2.size() ? i2[k - l] : 0;
        if (a != b)
            return a > b ? false : true;
    }

    return false;
}

TBigInt operator/(TBigInt i1, TBigInt i2)
{
    TBigInt ret = 0;

    if (i2.size() == 1) {
        int carry = 0;
        ret.Num.resize(i1.size(), 0);

        for (int k = i1.size() - 1; k >= 0; --k) {
            int t = i1[k] + carry * BASE;
            ret[k] = t / i2[0];
            carry = t % i2[0];
        }
    } else {
        int d = BASE / (i2[i2.size() - 1] + 1);
        int n = i2.size();
        int m;

        i1 = i1 * d;
        i2 = i2 * d;

        i1.Num.push_back(0);
        m = i1.size() - i2.size();
        
        if (m > 0) {
            ret.Num.resize(m, 0);
        }

        for(int j = m - 1; j >= 0; j--) {
            int q = (i1[j + n] * BASE + i1[j + n - 1]) / i2[n - 1];
            int r = (i1[j + n] * BASE + i1[j + n - 1]) % i2[n - 1];

            while (r < BASE && (q == BASE || (q * i2[n - 2]) > (BASE * r + i1[j + n - 2]))) {
                q--;
                r += i2[n - 1];
            }

            ret[j] = q;

            if (i1.LessThen(q * i2, j, j + n))
                ret[j]--;
        
            i1.Minus(ret[j] * i2, j, j + n);
        }
    }

    while(ret.Num.back() == 0 && ret.size() > 1)
        ret.Num.pop_back();

    return ret;
}

TBigInt pow(TBigInt val, int power)
{
    TBigInt ret;

    ret.Num.push_back(1);
    while (power > 0) {
        if ((power & 1) == 1)
            ret = ret * val;
        val = val * val;
        power >>= 1;
    }

    return ret;
}