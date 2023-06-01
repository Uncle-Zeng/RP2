"""
用于测试的主函数
注意：全部图像矩阵都是CHW，颜色通道为BGR（匹配torch的格式要求）
"""

import cv2
import numpy as np
import torch
import glob
import os

from utils import *

# 全局变量设置区域
args = get_default_configs()

# 原始图像 和 原始图像文件夹路径
original_img = get_original_img()
processed_img = preprocess_image(original_img)
original_img_dir = get_original_img_dir()

# 初始化的noise、生成的mask、以及二者生成的noise_mul
noise = generate_noise()
noise_mask = get_noise_mask()
# 利用noise和mask生成noise_mul，也就是被攻击的区域
noise_mul = torch.mul(noise, noise_mask)

# 最终输入模型的初始对抗样本，noise_input
noise_input = torch.mul(processed_img, noise_mul)
# 将大于1小于0的部分进行截断，得到最终的noise_input
noise_input[noise_input > 1.0] = 1.0
noise_input[noise_input < 0.0] = 0.0


# attack_target则是y，是目标类别的one-hot向量；adv_pred是模型输出的类别置信度向量
attack_target = get_attack_target()
adv_pred, label = model(noise_input)


# 用于测试现有的变量
def printtest():
    # 将图片进行水平拼接，从左至右分别是：noise，noise_mask，noise_mul，
    # original_image、processed_img（已标准化），noise_input
    horizontal_concat = np.hstack([noise.numpy().transpose(1, 2, 0),
                                   noise_mask.numpy().transpose(1, 2, 0),
                                   noise_mul.numpy().transpose(1, 2, 0),
                                   original_img / 255.0,
                                   processed_img.detach().numpy().transpose(1, 2, 0),
                                   noise_input.detach().numpy().transpose(1, 2, 0)])
    cv2.imshow("horizontal_concat", horizontal_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 其它打印信息
    # print(attack_target)
    # print(adv_pred)
    # print(label)


# 用来测试模型准确率
def modeltest():
    img_dir = "test_images/00001"
    img_list = glob.glob(os.path.join(img_dir, "*.png"))
    label_list = []
    counts = {}

    # 遍历文件夹中的所有png图片
    for img_path in img_list:
        img = cv2.imread(img_path)

        # 进行图像预处理
        img = transform(img)

        # 变换读入矩阵的维度信息

        pred, label = model(img)

        # 输出：每个图像的置信度和label；各个label计数，和预测准确率
        print(f"output:{pred.max():.3f}, class:{label.item()}, label:{classes[label.item()]}")
        # label计数
        label_list.append(label.item())

    # 遍历列表中的每个元素并进行统计
    for item in label_list:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1

    # 打印各个label计数，并计算预测准确率（counts[x]及时修改，与指定文件夹一致）
    print(f"各个class的计数 ：{counts}")
    print(f"pred_accuracy : {counts[1] / len(img_list):.3f}")


if __name__ == '__main__':
    # 输出置信度、类别、标签
    print(f"output : {adv_pred.max():.2f}\n"
          f"label : {label.item()}\n"
          f"class : {classes[label.item()]}")

    # 计算生成的初始对抗样本与原始图像之间的损失
    adversarial_loss = calculate_Loss(adv_pred)
    print(f"adversarial_loss = {adversarial_loss.item():.4f}")

    # 用于测试现有的变量
    printtest()

    # # 显示每张图像的l1mask
    # generate_noise_maskmatrix(2.5*10e-2, 5)

    # # 用来测试model分类效果的函数
    # modeltest()


