import os
import sys

import numpy as np
import imageio

in_path = sys.argv[1] if len(sys.argv) >= 2 else '.'
out_path = sys.argv[2] if len(sys.argv) >= 3 else './out'

with imageio.get_writer(f'{out_path}/movie.gif', mode='I', fps=20) as writer:
    num_img = len(os.listdir(f'{in_path}'))
    for filename in range(num_img):
        with open(f'{in_path}/{filename}.data', 'rb') as f:
            w, h = np.frombuffer(f.read(8), dtype=np.int32)
            img = np.frombuffer(f.read(), dtype=np.uint8)
            img = img.reshape((h, w, 4))
            writer.append_data(img)
