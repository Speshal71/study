import matplotlib.pyplot as plt
import sha1

def diff(n1, n2):
    cnt = 0
    diff = n1 ^ n2
    
    for i in range(diff.bit_length()):
        cnt += (diff >> i) & 1
    
    return cnt

def bytes_diff(arr1, arr2):
    return sum(diff(b1, b2) for b1, b2 in zip(arr1, arr2))


if __name__ == '__main__':
    text = input()

    original = bytearray(text, 'utf-8')

    modified = bytearray(text, 'utf-8')
    for i in range(len(modified)):
        modified[i] ^= 255

    rnd_cnt = []
    bits_diff = []

    for rnd in range(0, 80):
        rnd_cnt.append(rnd)
        bits_diff.append(bytes_diff(sha1.sha1(original, rnd),
                                    sha1.sha1(modified, rnd)))

    print('E = ', sum(bits_diff) / len(bits_diff))

    plt.plot(rnd_cnt, bits_diff)
    plt.xlabel('Номер раунда')
    plt.ylabel('Кол-во различных бит')
    plt.grid()
    plt.show()