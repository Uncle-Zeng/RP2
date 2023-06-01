"""
一些常用的函数
"""
import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from MyNet import Net
from configs import get_default_configs

# 变量设置区
args = get_default_configs()

# label和class对应字典
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}


# 预处理输入图像
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((32, 32)),
                                transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                                                     std=[0.2672, 0.2564, 0.2629])
                                ])


# transform平替版
# 对原始OpenCV读入的图像进行预处理（标准化、裁剪、变换维度、转换成tensor、求梯度）
def preprocess_image(image, resize=False):
    # 数据集颜色通道均值与标准差
    mean = [0.3337, 0.3064, 0.3171]
    std = [0.2672, 0.2564, 0.2629]

    # 对图片进行resize
    if resize:
        image = cv2.resize(image, (32, 32))

    # 转成float、将BGR转换成RGB、将HWC转换成CHW（为了方便输入模型）
    image = np.float32(image)
    # image = np.ascontiguousarray(image[..., ::-1])
    image = image.transpose(2, 0, 1)

    # 进行图片标准化计算
    for channel, _ in enumerate(image):
        image[channel] = image[channel] / 255
        image[channel] = image[channel] - mean[channel]
        image[channel] = image[channel] / std[channel]

    # 转换成tensor，并设置requires_grad=True
    image_tensor = torch.from_numpy(image)
    image_tensor.requires_grad_(True)

    return image_tensor


# 标准化的逆变换
def recreate_image(image):
    # 数据集逆变换的颜色通道均值与标准差
    reverse_mean = [-0.3337, -0.3064, -0.3171]
    reverse_std = [1/0.2672, 1/0.2564, 1/0.2629]
    recreated_image = copy.copy(image.data.numpy())

    # 对各个通道进行逆变换
    for c in range(3):
        recreated_image[c] /= reverse_std[c]
        recreated_image[c] -= reverse_mean[c]

    recreated_image[recreated_image > 1] = 1
    recreated_image[recreated_image < 0] = 0
    recreated_image = np.round(recreated_image * 255)

    recreated_image = np.uint8(recreated_image).transpose(1, 2, 0)
    # 将 RBG 转换为 GBR，方便opencv显示
    # recreated_image = recreated_image[..., ::-1]

    return recreated_image


# 用于分类的model，返回对抗样本输入后的模型预测的向量
def model(input):
    # 加载预训练的用分类GTSRB的网络，并加载参数
    mymodel = Net()
    mymodel.load_state_dict(torch.load("pretrained_models/model_40.pth"))

    # 输入图像为单张图像，增加一个batch维度
    # 注意：在对需要计算梯度的张量进行操作时，不能将其转换为 NumPy 数组，否则会导致计算图断裂，无法进行自动求导计算。
    # 但是不转换成numpy数组就不能输入
    input_img = input.unsqueeze(0)

    # 将模型切换到评估模式，并进行预测
    mymodel.eval()

    output = mymodel(input_img)
    output = torch.squeeze(output, dim=0)
    output = F.softmax(output, dim=0)
    label = output.argmax(dim=0, keepdim=True)

    return output, label


# 初始化noise颜色
def generate_noise():
    # 如果初始指定颜色不为空，则使用该颜色进行初始化
    if args.initial_value_for_noise != "" and args.initial_value_for_noise != " ":
        # 对参数进行分割，得到具体的rgb值并进行归一化，然后对逐个通道进行赋值
        noise_init_color = np.array(args.initial_value_for_noise.split(","), dtype=float) / 255.0
        r = noise_init_color[0]
        g = noise_init_color[1]
        b = noise_init_color[2]

        # 创建一个和输入图像大小相等的图片模板
        # 创建时就是符合PyTorch的要求，因此不需要进行维度转换
        noise_init = torch.rand(args.nb_channels, args.img_rows, args.img_cols)
        # 分别对每个通道进行颜色填充
        # 在处理图像时，PyTorch和OpenCV(cv2)默认都是使用BGR格式
        noise_init[0, :, :].fill_(b)
        noise_init[1, :, :].fill_(g)
        noise_init[2, :, :].fill_(r)

    # 如果初始指定颜色为空，则随机选取颜色进行初始化
    else:
        # 随机生成[0,1]之间的rgb值作为初始noise颜色
        noise_init = torch.rand(args.nb_channels, args.img_rows, args.img_cols)

    return noise_init


# 获取现有的mask
def get_noise_mask():
    noise_mask_dir = args.attack_mask
    noise_mask = cv2.imread(noise_mask_dir)

    # 转换为Tensor格式，并且转换维度信息（因为cv2读入的维度是HWC，而PyTorch一般要求CHW）
    # numpy变换维度二选一写法
    # noise_mask = np.transpose(noise_mask, (2, 0, 1))
    noise_mask = noise_mask.transpose(2, 0, 1)
    noise_mask = torch.from_numpy(noise_mask)

    return noise_mask / 255.0


# 用于生成每张图像的彩色与灰度L1梯度图，针对victim-set
def generate_noise_maskmatrix(size_threshold, number):
    """

    :param size_threshold: 阈值，用于控制二值图像的生成
    :param i: victim-set中第几个图像作为实验对象
    :return:
    """
    input_img = get_original_img()

    # 返回output和label，其中label用_替代
    output, _ = model(input_img)
    loss = calculate_Loss(output)
    loss.backward()

    matrix = input_img.grad

    # 获得原始梯度图像，对梯度矩阵进行min-max标准化
    matrix = matrix / (matrix.max() - matrix.min())

    # 生成灰度图像
    matrix_gray = torch.Tensor(args.img_rows, args.img_cols)
    for r in range(args.img_rows):
        for l in range(args.img_cols):
            matrix_gray[r, l] = 0.11 * matrix[0, r, l] + 0.3 * matrix[1, r, l] + 0.59 * matrix[2, r, l]

    # 生成0 1矩阵（可以通过调整阈值来实现对范围的控制）
    matrix_bool = torch.Tensor(args.img_rows, args.img_cols)
    # size_threshold = 2.0*10e-2
    for r in range(args.img_rows):
        for l in range(args.img_cols):
            # 要求绝对值大于阈值即可
            if abs(matrix_gray[r, l]) > size_threshold:
                matrix_bool[r, l] = 1
            else:
                matrix_bool[r, l] = 0

    # 显示梯度图像
    cv2.imshow("color", matrix.permute(1, 2, 0).numpy())
    cv2.imshow("gray", matrix_gray.numpy())
    cv2.imshow("bool", matrix_bool.numpy())

    # 写入梯度图像，i为victim中的序号
    i = number
    savefile1 = "L1_gradmap/size_threshold=2.5/" + str(i) + "matrix.png"
    savefile2 = "L1_gradmap/size_threshold=2.5/" + str(i) + "matrix_gray.png"
    savefile3 = "L1_gradmap/size_threshold=2.5/" + str(i) + "matrix_bool.png"
    cv2.imwrite(savefile1, matrix.permute(1, 2, 0).numpy()*255)
    cv2.imwrite(savefile2, matrix_gray.numpy()*255)
    cv2.imwrite(savefile3, matrix_bool.numpy()*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 计算NPS值，NPS是为了尽可能地使用打印机拥有的颜色，减少打印失真
# 计算的最终生成的“贴图”与可打印颜色之间的差别
def calculate_NPS(noise_mul):
    noise_mul = noise_mul.double()
    NPSs = torch.ones_like(noise_mul)
    # 可打印颜色集合获取  torch.Size([30, 32, 32, 3])
    printable_colors = get_print_triplets(args.printability_tuples)
    for i in range(noise_mul.size(0)):
        for j in range(noise_mul.size(1)):
            for k in range(printable_colors.size(0)):
                printable_color = printable_colors[k, :]
                NPSs[i, j] = NPSs[i, j] * \
                             torch.sum(torch.square(torch.sub(noise_mul[i, j, :], printable_color[i, j, :])))

    NPS = torch.sum(NPSs)

    return NPS


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
def calculate_Loss(output):
    # 用于计算损失函数的变量
    attack_target = get_attack_target()
    noise_mask = get_noise_mask()
    noise = generate_noise()
    noise_mul = torch.mul(noise, noise_mask)

    # 如果使用NPS项，则加上calculate_NPS()
    if args.printability_optimization:
        RP2_loss = calculate_OptimizationLoss(args.optimization_loss, attack_target, output) + \
                   calculate_RegLoss(args.regloss, noise_mask, noise) + calculate_NPS(noise_mul)
    # 如果不使用NPS项，则仅有两项
    else:
        RP2_loss = calculate_OptimizationLoss(args.optimization_loss, attack_target, output) + \
                   calculate_RegLoss(args.regloss, noise_mask, noise)

    return RP2_loss


# 在指定的路径下获取用于打印优化的文件数据（三元组）
def get_print_triplets(src):
    p = []
    # load the triplets and create an array of the speified size
    with open(args.printability_tuples) as f:
        for line in f:
            p.append(line.split(","))
    # 对p中的每个元素都进行复制，生成和img_cols,img_rows等大的图片列表
    p = map(lambda x: [[x for _ in range(args.img_cols)] for __ in range(args.img_rows)], p)
    p = list(p)
    p = torch.from_numpy(np.float32(p))

    return p


# 返回原始图片的tensor值
def get_original_img():
    img_path = args.attack_srcimg
    img = cv2.imread(img_path)

    # img2 = cv2.imread("L1_gradmap/size_threshold=2.0/3matrix_bool.png") / 255.0
    # # 将矩阵中的 0 和 1 值互换
    # img2 = np.where(np.equal(img2, 0), 1, 0)
    #
    # img = np.multiply(img, img2)

    # 转成float才能设置requires_grad = True
    # 这里的输入图像没有进行维度变换，为什么可以与其它图片一起做变换呢？因为经过了transform
    # img = transform(img).float()
    # img.requires_grad_(True)

    return img


# 返回原始图片所在文件夹路径
def get_original_img_dir():

    return args.attack_srcdir


# attack_target则是y，是目标类别的one-hot向量；
def get_attack_target():
    one_hot = torch.eye(args.nb_classes)[args.target_class]

    return one_hot

