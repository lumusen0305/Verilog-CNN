
import numpy as np
def save_hex_array_formatted(hex_array, filename):
    with open(filename, 'w') as f:
        if hex_array.ndim == 1:
            # 处理一维数组
            row_str = ' '.join(hex_array)
            f.write(row_str + '\n')
        elif hex_array.ndim == 2:
            # 处理二维数组
            for i in range(hex_array.shape[0]):
                row = hex_array[i]
                row_str = ' '.join(row)
                f.write(row_str + '\n')
        elif hex_array.ndim == 3:
            # 处理三维数组
            for i in range(hex_array.shape[0]):
                f.write(f"Layer {i}:\n")
                for j in range(hex_array.shape[1]):
                    row = hex_array[i, j]
                    row_str = ' '.join(row)
                    f.write('  ' + row_str + '\n')
                f.write('\n')
        elif hex_array.ndim == 4:
            # 处理四维数组
            for i in range(hex_array.shape[0]):
                f.write(f"Filter {i}:\n")
                for j in range(hex_array.shape[1]):
                    f.write(f"  Channel {j}:\n")
                    for k in range(hex_array.shape[2]):
                        row = hex_array[i, j, k]
                        row_str = ' '.join(row)
                        f.write('    ' + row_str + '\n')
                    f.write('\n')
                f.write('\n')
        else:
            raise ValueError(f"Unsupported array dimension: {hex_array.ndim}")
def float_to_hex16(value):
    """
    将浮点数转换为16位的十六进制字符串表示，前8位整数，后8位小数。

    参数:
        value: 浮点数或包含浮点数的NumPy数组

    返回:
        对应的16位十六进制字符串或字符串数组
    """
    # 确保输入是NumPy数组
    value = np.array(value, dtype=np.float32)
    
    # 创建一个函数用于处理单个浮点数
    def convert_single(val):
        # 提取整数部分和小数部分
        int_part = int(np.floor(val))
        frac_part = val - int_part
        
        # 处理整数部分，限制在 -128 到 127
        if int_part < -128:
            int_part = -128
        elif int_part > 127:
            int_part = 127
        
        # 将整数部分转换为有符号8位整数的二进制表示
        int_byte = int_part & 0xFF  # 取低8位
        # 如果是负数，进行补码转换
        if int_part < 0:
            int_byte = (256 + int_part) & 0xFF
        
        # 将小数部分转换为 0 - 255 的整数
        frac_byte = int(round(frac_part * 256)) & 0xFF
        # 特殊处理，当 frac_part == 1.0 时
        if frac_byte == 256:
            frac_byte = 255
        
        # 合并整数部分和小数部分
        hex_str = f"{int_byte:02X}{frac_byte:02X}"
        return hex_str
    
    # 对数组中的每个元素进行转换
    if value.ndim == 0:
        # 单个数值
        return convert_single(value)
    else:
        # 数组，逐元素转换
        vectorized_convert = np.vectorize(convert_single)
        hex_array = vectorized_convert(value)
        return hex_array

def read_mnist_txt_file(filename):
    with open(filename, 'r') as f:
        content = f.read().strip()
    # 将所有值拆分为列表
    values = content.split()
    # 将十六进制字符串转换为整数
    values = [int(val, 16) for val in values if val.strip() != '']
    # 转换为 numpy 数组
    image = np.array(values, dtype=np.float32)
    # 检查是否有 28*28 = 784 个像素值
    if image.size != 28 * 28:
        print(f"Error: The image contains {image.size} pixels, expected 784.")
        return None
    # 重塑为 (28, 28)
    image = image.reshape((28, 28))
    # 归一化到 [0, 1]
    # image = image / 255.0
    # 标准化
    # image = (image - 0.1307) / 0.3081
    return image            
# 读取保存的参数
data_name="9_0.txt"
# 从 txt 文件读取 MNIST 图像
image = read_mnist_txt_file(data_name)  #为你的文件名

# 归一化图像
image = (image - 0.1307) / 0.3081

# 添加通道维度
input_image = image[np.newaxis, :, :]  # 形状为 (1, 28, 28)
save_hex_array_formatted(float_to_hex16(input_image),data_name)

