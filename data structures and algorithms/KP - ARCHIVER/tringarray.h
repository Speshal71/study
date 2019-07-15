#ifndef TRINGARRAY_H
#define TRINGARRAY_H

template<typename T> struct TRingArray
{
    T *arr;
    int size, cap, pos;

    TRingArray(int _cap): cap(_cap), size(0), pos(0)
    {
        arr = new T[cap];
    }

    int RealIndex(int index)
    {
        int newIndex = index % cap;
        return newIndex + (newIndex >= 0 ? 0 : cap);
    }

    T& operator[](int index)
    {
        return arr[RealIndex(index)];
    }

    void Push(T el)
    {
        arr[(pos + size) % cap] = el;

        if (size < cap) {
            ++size;
        } else {
            pos = ++pos % cap;
        }
    }
    
    void PopFront()
    {
        pos = ++pos % cap;
        --size;
    }

    int EndPos()
    {
        return (pos + size - 1) % cap;
    }

    int DistanceToEnd(int from)
    {
        int end = (pos + size - 1) % cap;
        return end - from + 1 + (end < from ? size : 0);
    }

    ~TRingArray()
    {
        delete[] arr;
    }
};

#endif