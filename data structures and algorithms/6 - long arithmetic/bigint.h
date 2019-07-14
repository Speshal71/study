#ifndef BIGINT_H
#define BIGINT_H

#include <iostream>
#include <vector>

const int BASE = 10000;
const int BASELEN = 4;

class TBigInt
{
private:
    std::vector<int> Num;

    inline int& operator[](int k);
    inline int operator[](int k) const;
    inline size_t size() const;

    TBigInt& Minus(const TBigInt& i2, int l, int r);
    TBigInt& Plus(const TBigInt& i2, int l, int r);
    bool LessThen(const TBigInt& i2, int l, int r);
public:
    TBigInt() {};
    TBigInt(int i);

    int ToInt();

    friend std::istream& operator>>(std::istream& is, TBigInt& i);
    friend std::ostream& operator<<(std::ostream& os, const TBigInt& i);
    friend bool operator==(const TBigInt& i1, const TBigInt& i2);
    friend bool operator>(const TBigInt& i1, const TBigInt& i2);
    friend bool operator<(const TBigInt& i1, const TBigInt& i2);
    friend TBigInt operator+(const TBigInt& i1, const TBigInt& i2);
    friend TBigInt operator-(const TBigInt& i1, const TBigInt& i2);
    friend TBigInt operator*(const TBigInt& i1, const TBigInt& i2);
    friend TBigInt operator/(TBigInt i1, TBigInt i2);
    friend TBigInt pow(TBigInt val, int power);
};

#endif 