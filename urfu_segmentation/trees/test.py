from PIL import Image
import numpy as np

file_path = '/misc/home6/m_imm_freedata/Segmentation/Projects/mmseg_trees/RG2/images/0.tif'

# Открываем изображение
img = Image.open(file_path)

# Конвертируем в numpy array
img_np = np.array(img)

# Выводим shape
print("Shape:", img_np.shape)
print("Shape:", np.unique(img_np, axis=-1))