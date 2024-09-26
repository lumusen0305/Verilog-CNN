import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

def hex32_to_float(hex_value):
    """
    将32位定点数十六进制字符串（Q16.16格式）转换为浮点数。

    参数:
        hex_value: 包含32位定点数十六进制字符串的数组或单个字符串

    返回:
        对应的浮点数或浮点数数组
    """
    # 确保输入是NumPy数组
    hex_value = np.array(hex_value)
    
    # 创建一个函数用于处理单个十六进制字符串
    def convert_single(hex_str):
        # 解析十六进制字符串为整数
        full_int = int(hex_str, 16)
        
        # 处理补码负数
        if full_int >= 0x80000000:
            full_int -= 0x100000000
        
        # 提取整数部分和小数部分
        int_part = full_int >> 16  # 取高16位
        frac_part = full_int & 0xFFFF  # 取低16位
        
        # 将小数部分转换为浮点数
        frac_float = frac_part / 65536.0
        
        # 组合整数部分和小数部分
        result = int_part + frac_float
        return result
    
    # 对数组中的每个元素进行转换
    if hex_value.ndim == 0:
        # 单个数值
        return convert_single(hex_value)
    else:
        # 数组，逐元素转换
        vectorized_convert = np.vectorize(convert_single)
        float_array = vectorized_convert(hex_value)
        return float_array
def hex16_to_float(hex_array):
    """
    将 Q8.8 格式的十六进制字符串转换为浮点数。

    参数:
        hex_array: 包含 Q8.8 十六进制字符串的数组或列表
    
    返回:
        转换后的浮点数数组
    """
    def convert_single(hex_str):
        # 将十六进制字符串转换为 16 位有符号整数
        int_value = int(hex_str, 16)
        # 处理负数，Q8.8 格式是有符号数，检查符号位
        if int_value > 0x7FFF:
            int_value -= 0x10000
        
        # Q8.8 定点数格式，转换回浮点数，除以 256
        return int_value / 256.0

    # 将数组中的每个十六进制字符串逐个转换
    vectorized_convert = np.vectorize(convert_single)
    return vectorized_convert(hex_array)
def float_to_hex32(value):
    """
    将浮点数转换为32位的十六进制字符串表示，前16位整数，后16位小数。

    参数:
        value: 浮点数或包含浮点数的NumPy数组

    返回:
        对应的32位十六进制字符串或字符串数组
    """
    # 确保输入是NumPy数组
    value = np.array(value, dtype=np.float32)
    
    # 创建一个函数用于处理单个浮点数
    def convert_single(val):
        # 提取整数部分和小数部分
        int_part = int(np.floor(val))
        frac_part = val - int_part
        
        # 处理整数部分，限制在 -32768 到 32767
        if int_part < -32768:
            int_part = -32768
        elif int_part > 32767:
            int_part = 32767
        
        # 将整数部分转换为有符号16位整数的二进制表示
        int_byte = int_part & 0xFFFF  # 取低16位
        # 如果是负数，进行补码转换
        if int_part < 0:
            int_byte = (65536 + int_part) & 0xFFFF
        
        # 将小数部分转换为 0 - 65535 的整数
        frac_byte = int(round(frac_part * 65536)) & 0xFFFF
        # 特殊处理，当 frac_part == 1.0 时
        if frac_byte == 65536:
            frac_byte = 65535
        
        # 合并整数部分和小数部分
        hex_str = f"{int_byte:04X}{frac_byte:04X}"
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
# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 第一层卷积层
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        # 最大池化层
        self.mp = nn.MaxPool2d(2)
        # 第二层卷积层
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5)
        # 全连接层
        self.fc_1 = nn.Linear(48, 10)
        
        # 存储中间输出（可选）
        self.conv1_out_np = np.zeros((1, 3, 24, 24))
        self.mp1_out_np = np.zeros((1, 3, 12, 12))
        self.conv2_out_np = np.zeros((1, 3, 8, 8))
        self.mp2_out_np = np.zeros((1, 3, 4, 4))
        self.fc_in_np = np.zeros((1, 48))
        self.fc_out_np = np.zeros((1, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        self.conv1_out_np = x.detach().cpu().numpy()
        
        x = self.mp(x)
        self.mp1_out_np = x.detach().cpu().numpy()
        
        x = self.conv2(x)
        x = F.relu(x)
        self.conv2_out_np = x.detach().cpu().numpy()
        
        x = self.mp(x)
        self.mp2_out_np = x.detach().cpu().numpy()
        
        x = x.view(-1, 48)
        self.fc_in_np = x.detach().cpu().numpy()
        
        x = self.fc_1(x)
        self.fc_out_np = x.detach().cpu().numpy()
        
        return F.log_softmax(x, dim=1)

# 训练参数设置
batch_size = 128
epochs = 6
learning_rate = 0.01
momentum = 0.5
log_interval = 100

# 检查 GPU 可用性                                   
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=1, pin_memory=True)

# 初始化模型、优化器和损失函数
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('训练周期: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n测试集: 平均损失: {:.4f}, 准确率: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 训练和测试模型
# for epoch in range(1, epochs +1):
#     train(model, device, train_loader, optimizer, epoch)
#     test(model, device, test_loader)

# 保存训练好的权重和偏置
# torch.save(model.state_dict(), 'cnn_mnist.pth')
state_dict = torch.load('cnn_mnist.pth')
model.load_state_dict(state_dict)
# 提取权重和偏置
params = {}
for name, param in model.named_parameters():
    params[name] = param.detach().cpu().numpy()

# 保存参数
np.save('params.npy', params)

# 手动实现前向传播
def conv2d_manual(input, weight, bias):
    in_channels, H_in, W_in = input.shape
    out_channels, _, kernel_size, _ = weight.shape
    H_out = H_in - kernel_size + 1
    W_out = W_in - kernel_size + 1
    output = np.zeros((out_channels, H_out, W_out))
    for out_c in range(out_channels):
        for i in range(H_out):
            for j in range(W_out):
                sum = 0.0
                for in_c in range(in_channels):
                    sum += np.sum(
                        input[in_c, i:i+kernel_size, j:j+kernel_size] * weight[out_c, in_c])
                sum += bias[out_c]
                output[out_c, i, j] = sum
    return output

def relu_manual(x):
    return np.maximum(0, x)

def maxpool2d_manual(input, kernel_size, stride):
    channels, H_in, W_in = input.shape
    H_out = (H_in - kernel_size) // stride + 1
    W_out = (W_in - kernel_size) // stride + 1
    output = np.zeros((channels, H_out, W_out))
    for c in range(channels):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + kernel_size
                w_start = j * stride
                w_end = w_start + kernel_size
                output[c, i, j] = np.max(input[c, h_start:h_end, w_start:w_end])
    return output

def linear_manual(input, weight, bias):
    output = np.dot(weight, input) + bias
    return output

def softmax_manual(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
params = np.load('params.npy', allow_pickle=True).item()

# 从 txt 文件读取 MNIST 图像
image = read_mnist_txt_file('3_0.txt')  #为你的文件名

# 归一化图像
image = (image - 0.1307) / 0.3081

# 添加通道维度
input_image = image[np.newaxis, :, :]  # 形状为 (1, 28, 28)
save_hex_array_formatted(float_to_hex32(input_image),"input_image.txt")
input_image=hex32_to_float(float_to_hex32(input_image))
# 手动前向传播
# 第一层卷积
conv1_weight = params['conv1.weight']
conv1_bias = params['conv1.bias']
print("Loaded Conv1 Weights:", float_to_hex32(conv1_weight))
print("Loaded Conv1 Bias:", float_to_hex32(conv1_bias                       ))
save_hex_array_formatted(float_to_hex32(conv1_weight),"conv1_weight.txt")
# print("Loaded Conv1 Biases:", float_to_hex32(conv1_bias))  
save_hex_array_formatted(float_to_hex32(conv1_bias),"conv1_bias.txt")
conv1_weight =hex32_to_float(float_to_hex32(conv1_weight))
conv1_bias   =hex32_to_float(float_to_hex32(conv1_bias))
conv1_out = conv2d_manual(input_image, conv1_weight, conv1_bias)

# conv1_out = relu_manual(conv1_out)
# print(conv1_out)
# 第一次最大池化
print("Loaded conv1_out:", float_to_hex32(conv1_out[0]))  
# print("Loaded conv2_out:", float_to_hex32(conv2_out[0]))  

mp1_out = relu_manual(maxpool2d_manual(conv1_out, kernel_size=2, stride=2))
print("Loaded mp1_out:", float_to_hex32(mp1_out[0]))  


# 第二层卷积
conv2_weight = params['conv2.weight']               
conv2_weight = hex32_to_float(float_to_hex32(conv2_weight))
conv2_bias = params['conv2.bias']
conv2_bias = hex32_to_float(float_to_hex32(conv2_bias))

print("Loaded Conv2 Weights:", float_to_hex32(conv2_weight))
save_hex_array_formatted(float_to_hex32(conv2_weight),"conv2_weight.txt")
print("Loaded Conv2 Biases:", float_to_hex32(conv2_bias))   
save_hex_array_formatted(float_to_hex32(conv2_bias),"conv2_bias.txt")

conv2_out = conv2d_manual(mp1_out, conv2_weight, conv2_bias)

# conv2_out = relu_manual(conv2_out)
print("Loaded conv2_out:", float_to_hex32(conv2_out[0]))  

# 第二次最大池化
mp2_out = relu_manual(maxpool2d_manual(conv2_out, kernel_size=2, stride=2))
print("Loaded mp2_out:", float_to_hex32(mp2_out[0]))  

# 展平
fc_in = mp2_out.reshape(-1)

# 全连接层
fc_weight = params['fc_1.weight']
fc_bias = params['fc_1.bias']
fc_weight = hex32_to_float(float_to_hex32(fc_weight))
fc_bias = hex32_to_float(float_to_hex32(fc_bias))

print("Loaded FC Weights:", float_to_hex32(fc_weight))
save_hex_array_formatted(float_to_hex32(fc_weight),"fc_weight.txt")


print("Loaded FC Biases:", float_to_hex32(fc_bias))   
save_hex_array_formatted(float_to_hex32(fc_bias),"fc_bias.txt")

fc_out = linear_manual(fc_in, fc_weight, fc_bias)

# Softmax 输出
output_probs = softmax_manual(fc_out)

# 预测类别
predicted_class = np.argmax(output_probs)

print('预测的类别:', predicted_class)
print('输出概率:', output_probs)    

