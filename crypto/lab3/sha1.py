MAX_ROUND  = 80
BLOCK_SIZE = 64

def rotate_left(num, k):
        return ((num << k) | (num >> (32 - k))) & 0xffffffff

def hash_block(block, h, rnd=MAX_ROUND):
    w = [0] * 80

    for i in range(0, 16):
        w[i] = int.from_bytes(block[4 * i: 4 * i + 4], byteorder='big')

    for i in range(16, 80):
        w[i] = rotate_left(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1)

    a, b, c, d, e = h
      
    for i in range(rnd):
        if 0 <= i <= 19:
            f = d ^ (b & (c ^ d))
            k = 0x5A827999
        elif 20 <= i <= 39:
            f = b ^ c ^ d
            k = 0x6ED9EBA1
        elif 40 <= i <= 59:
            f = (b & c) | (b & d) | (c & d)
            k = 0x8F1BBCDC
        elif 60 <= i <= 79:
            f = b ^ c ^ d
            k = 0xCA62C1D6

        temp = (rotate_left(a, 5) + f + e + w[i] + k) & 0xffffffff
        e = d
        d = c
        c = rotate_left(b, 30)
        b = a
        a = temp
    
    h = ((h[0] + a) & 0xffffffff,
         (h[1] + b) & 0xffffffff,
         (h[2] + c) & 0xffffffff,
         (h[3] + d) & 0xffffffff,
         (h[4] + e) & 0xffffffff)
    
    return  h

def sha1(bstr, round_cnt=MAX_ROUND):
    h = (0x67452301,
         0xEFCDAB89,
         0x98BADCFE,
         0x10325476,
         0xC3D2E1F0)
    
    if round_cnt > MAX_ROUND:
        round_cnt = MAX_ROUND

    old_len = len(bstr) * 8
    bstr += b'\x80'
    while len(bstr) % 64 != 60:
        bstr += b'\x00'
    bstr += old_len.to_bytes(4, byteorder='big')
    
    for i in range(0, len(bstr), BLOCK_SIZE):
        block = bstr[i: i + BLOCK_SIZE]
        h = hash_block(block, h, rnd=round_cnt)
    
    
    ret = b''
    
    for i in h:
        ret += i.to_bytes(4, byteorder='big')
    
    return ret