from pygost import gost34112012256

def get_algo(name):
    bytes_name = bytes(name, 'utf-8')
    hash_name = gost34112012256.new(bytes_name).digest()
    return hash_name[-1] & 15
