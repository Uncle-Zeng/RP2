"""
加入 mask 和 损失函数 的目标攻击算法
尝试使用其它优化算法进行攻击...
"""

from utils import *
from torch.optim import SGD


# 计算正则化损失
def calculate_RegLoss(judge, noise_mask, noise):
    # 使用l1范数进行计算
    if judge:
        RegLoss = args.attack_lambda * torch.norm(torch.mul(noise_mask, noise), p=1)

    # 使用l2范数进行计算
    else:
        RegLoss = args.attack_lambda * torch.norm(torch.mul(noise_mask, noise), p=2)

    return RegLoss


# 计算优化损失函数损失
def calculate_OptimizationLoss(judge, attack_target, adv_pred):
    # 使用MSE函数进行计算损失
    if judge:
        loss = torch.nn.MSELoss()
        OptimizationLoss = loss(attack_target, adv_pred)

    # 使用交叉熵损失函数进行计算损失
    else:
        loss = torch.nn.CrossEntropyLoss()
        OptimizationLoss = loss(attack_target, adv_pred)

    return OptimizationLoss


# 计算整个损失函数，其中NPS项可选
def calculate_Loss(target, output, noise_mask, noise):

    RP2_loss = calculate_OptimizationLoss(args.optimization_loss, target, output) + \
               calculate_RegLoss(args.regloss, noise_mask, noise)

    return RP2_loss


class FastGradientSigntargeted:
    def __init__(self, alpha):
        # 设置学习率
        self.alpha = alpha

    #  输入原始图像和原始图像的ID
    def generate(self, image, ori_class, tar_class):
        # 转换成tensor张量便于计算
        tar_class_tensor = torch.eye(43)[tar_class]

        adv_noise = torch.ones_like(mask_tensor)

        # 图像预处理
        processed_image = preprocess_image(image)

        # 开始迭代，尝试在1000次迭代中成功攻击
        for i in range(1000):
            # print('Iteration:', str(i + 1))
            # 梯度清零
            processed_image.grad = None
            # 前向传播
            out, _ = model(processed_image)
            # 计算损失模型损失
            pred_loss = calculate_Loss(tar_class_tensor, out, mask_tensor, adv_noise)
            print(f"iter:{str(i + 1)}, loss={pred_loss.item():.3f}")
            # 反向传播
            pred_loss.backward()

            # 生成噪声，并利用mask实现局部扰动
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            final_noise = torch.mul(adv_noise, mask_tensor)

            # 将对抗扰动添加到预处理后的图像中
            processed_image.data = processed_image.data - final_noise
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
    target_class = 4

    # 原始图像并进行预处理
    original_image = cv2.imread("victim-set/5.png", 1)

    # 读取mask，转换成tensor并转换维度
    mask = cv2.imread("L1_gradmap/size_threshold=1.5/5matrix_bool.png", 1)
    mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1)) / 255.0

    # 实例化对象，学习率为0.01
    FGS_targeted = FastGradientSigntargeted(0.01)

    # 输入：原始图像、原始类别、目标类别
    FGS_targeted.generate(original_image, original_class, target_class)
