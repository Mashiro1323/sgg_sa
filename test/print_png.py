from PIL import Image
import numpy as np

# 打开图像
image_path = 'test\exp_sa.png'
image = Image.open(image_path)

# 确保图像是单通道
if image.mode != 'L':
    raise ValueError("The image is not in single channel (grayscale) format.")

# 将图像转换为 numpy 数组
image_array = np.array(image)
output_path_text = 'test\image_array.txt'
# np.savetxt(output_path_text, image_array, fmt='%d')
print(np.zeros((5,3)))

# 打印矩阵值
# print(image_array)

