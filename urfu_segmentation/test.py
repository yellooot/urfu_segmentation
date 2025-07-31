from PIL import Image
import numpy as np


path = 'visualized_18450163/pred/00000000_pred.png'

im_frame = Image.open(path)
np_frame = np.array(im_frame.getdata())
print(np_frame.max(), np_frame.min(), set(np_frame.tolist()), np_frame.shape)