import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

# 定义 CNN 模型
class QuantizedCNN(nn.Module):
    def __init__(self):
        super(QuantizedCNN, self).__init__()
        
        # 添加量化/反量化模块
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # 第一层卷积层
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2)
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc_1 = nn.Linear(48, 10)
        
    def forward(self, x):
        # 量化输入
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        
        x = x.view(-1, 48)
        x = self.fc_1(x)
        
        # 反量化输出
        x = self.dequant(x)
        
        return F.log_softmax(x, dim=1)

# 训练参数设置
batch_size = 64
epochs = 2
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
model = QuantizedCNN().to(device)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 准备量化  
torch.quantization.prepare(model, inplace=True)

# 使用训练数据对模型进行校准
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            model(data)

# 对模型进行校准
calibrate(model, train_loader)

# 转换为量化模型
torch.quantization.convert(model, inplace=True)
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
for epoch in range(1, epochs +1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 保存训练好的权重和偏置
torch.save(model.state_dict(), 'cnn_mnist.pth')

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
image = read_mnist_txt_file('4_0.txt')  #       为你的文件名

# 归一化图像
image = (image - 0.1307) / 0.3081

# 添加通道维度
input_image = image[np.newaxis, :, :]  # 形状为 (1, 28, 28)

# 手动前向传播
# 第一层卷积
conv1_weight = params['conv1.weight']
conv1_bias = params['conv1.bias']
print("Loaded Conv1 Weights:", conv1_weight)
print("Loaded Conv1 Biases:", conv1_bias)       
conv1_out = conv2d_manual(input_image, conv1_weight, conv1_bias)
conv1_out = relu_manual(conv1_out)
# print(conv1_out)
# 第一次最大池化
mp1_out = maxpool2d_manual(conv1_out, kernel_size=2, stride=2)

# 第二层卷积
conv2_weight = params['conv2.weight']
conv2_bias = params['conv2.bias']
print("Loaded Conv2 Weights:", conv2_weight)
print("Loaded Conv2 Biases:", conv2_bias)   
conv2_out = conv2d_manual(mp1_out, conv2_weight, conv2_bias)
conv2_out = relu_manual(conv2_out)
    


# 第二次最大池化
mp2_out = maxpool2d_manual(conv2_out, kernel_size=2, stride=2)

# 展平
fc_in = mp2_out.reshape(-1)

# 全连接层
fc_weight = params['fc_1.weight']
fc_bias = params['fc_1.bias']
print("Loaded FC Weights:", fc_weight)
print("Loaded FC Biases:", fc_bias  )   

fc_out = linear_manual(fc_in, fc_weight, fc_bias)

# Softmax 输出
output_probs = softmax_manual(fc_out)

# 预测类别
predicted_class = np.argmax(output_probs)

print('预测的类别:', predicted_class)
print('输出概率:', output_probs)