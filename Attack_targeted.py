"""
最简单的目标攻击算法
"""

import warnings

from torch import nn

from utils import *

warnings.filterwarnings("ignore", category=UserWarning)


class FastGradientSigntargeted:
    def __init__(self, alpha):
        # 设置学习率
        self.alpha = alpha

    #  输入原始图像和原始图像的ID
    def generate(self, image, ori_class, tar_class):
        # 转换成tensor张量便于计算
        tar_class_tensor = torch.eye(43)[tar_class]

        # 定义损失函数
        loss = nn.CrossEntropyLoss()

        # 图像预处理
        processed_image = preprocess_image(image)

        # 开始迭代，尝试在10次迭代中成功攻击
        for i in range(1000):
            # print('Iteration:', str(i + 1))
            # 梯度清零
            processed_image.grad = None
            # 前向传播
            out, _ = model(processed_image)
            # 计算损失模型损失
            pred_loss = loss(out, tar_class_tensor)
            print(f"iter:{str(i+1)}, loss={pred_loss.item():.3f}")
            # 反向传播
            pred_loss.backward()
            # 生成噪声，全局扰动
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # 将对抗扰动添加到预处理后的图像中
            processed_image.data = processed_image.data - adv_noise
            # 生成合成图像
            recreated_image = recreate_image(processed_image)
            # 对合成图像进进行处理
            prep_confirmation_image = preprocess_image(recreated_image)
            # 确认图像的前向传播
            confirmation_output, confirmation_label = model(prep_confirmation_image)
            # 获取预测置信度
            confirmation_confidence = confirmation_output.max()

            # 判断是否成功攻击，且是否达到预期的置信度
            if confirmation_label == tar_class and confirmation_confidence >= 0.9:
                print(f"Original image : {ori_class} {classes[ori_class]}")
                print(f"Adversarial image : {confirmation_label.item()} {classes[confirmation_label.item()]}")
                print(f"And predicted with confidence of : {confirmation_confidence.item():.3f}")

                noise_image = recreated_image - image

                # 显示图像（原始图像，噪声，生成图像）
                cv2.imshow("original_image", original_image)
                cv2.imshow("noise_image", noise_image)
                cv2.imshow("recreated_image", recreated_image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                break
        return 1


if __name__ == '__main__':
    # 原始类别和目标类别
    original_class = 14
    target_class = 2

    # 原始图像并进行预处理
    original_image = cv2.imread("victim-set/5.png", 1)
    # original_image = preprocess_image(original_image)

    # 实例化对象，学习率为0.01
    FGS_targeted = FastGradientSigntargeted(0.01)

    # 输入：原始图像、原始类别、目标类别
    FGS_targeted.generate(original_image, original_class, target_class)


