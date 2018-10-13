import json
import numpy as np
from PIL import Image


class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        self.f_log = open(path, 'w')
        self.f_log.write(json.dumps(kwargs) + '\n')

    def log(self, **kwargs):
        self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()


def npy2img(filename, x):
    x = x.astype(np.uint8)
    n_rows = 10
    n_cols = 10
    imgs = list(map(Image.fromarray, x[0:n_rows*n_cols, :, :, :]))
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths) // n_cols
    total_height = sum(heights) // n_rows

    new_img = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i*10 + j
            new_img.paste(imgs[idx], (x_offset, y_offset))
            x_offset += imgs[idx].size[0]
        y_offset += imgs[idx].size[0]
        x_offset = 0
    new_img.save('{}.png'.format(filename))
