# 用于参数设置

import argparse


def get_default_configs():
    # 使用python内置的argparse来进行参数设置，也方便调参
    # 使用ArgumentParser对象进行初始化
    parser = argparse.ArgumentParser()

    # 参数定义
    # -- 定义了参数的名称、type 规定了参数的格式、default 规定了默认值、help为该参数的简单说明

    # 图片相关
    # 预设的输入图像的row、col、channel
    parser.add_argument('--img_rows', type=int, default=32, help='Input row dimension')
    parser.add_argument('--img_cols', type=int, default=32, help='Input column dimension')
    parser.add_argument('--nb_channels', type=int, default=3, help='number of color channels in the input.')
    # 原始图像的分类类别个数，用于生成one-hot向量；攻击后的图像类别编号
    parser.add_argument('--nb_classes', type=int, default=43, help='Number of classification classes')
    parser.add_argument('--target_class', type=int, default=14,
                        help='The class being targeted for this attack as a number')

    # 优化器相关
    parser.add_argument('--optimization_rate', type=float, default=0.01, help='The learning rate for the Adam '
                                                                              'optimizer of the attack objective')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='The beta1 parameter for the Adam optimizer of '
                                                                      'the attack objective')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='The beta2 parameter for the Adam optimizer '
                                                                        'of the attack objective')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help='The epsilon parameter for the Adam '
                                                                          'optimizer of the attack objective')
    # 构建损失函数相关
    parser.add_argument('--printability_optimization', type=bool, default=False,
                        help='是否使用NPS. True：使用, False：不使用')
    parser.add_argument('--optimization_loss', type=bool, default=False,
                        help='损失函数二选一。True：mes, False：cross_entropy')
    # 仅当生成mask时采用L1范数，其它场景均采用L2范数
    parser.add_argument('--regloss', type=bool, default=False, help='正则化项的范数选择。True：l1, False：l2')
    parser.add_argument('--attack_lambda', type=float, default=0.02, help='正则化项前的超参数')

    # 路径相关
    # 打印优化数据的路径
    parser.add_argument('--printability_tuples', type=str, default='npstriplets.txt', help='打印机可打印颜色')
    # 被攻击的图像具体路径、文件夹路径
    parser.add_argument('--attack_srcimg', type=str, default='victim-set/3.png', help='Filepath to image to mount the '
                                                                                      'attack on, if running a '
                                                                                      'one-image script.')
    parser.add_argument('--attack_srcdir', type=str, default='victim-set', help='Filepath to the directory containing '
                                                                                'the images to train on, if running a'
                                                                                ' multi-image script.')
    # 用于攻击的mask路径
    parser.add_argument('--attack_mask', type=str, default='mask/octagon.png', help='Filepath to the mask used in the '
                                                                                    'attack.')
    # model路径
    parser.add_argument('--model_path', type=str, default='models/all_r_ivan.ckpt', help='Path to load model from.')

    # 关于noise初始颜色生成
    # 红色：255 0 0 绿色：0 255 0  蓝色：0 0 255
    parser.add_argument('--initial_value_for_noise', type=str, default='255,255,255', help='noise的初始颜色，rgb值')

    # 使用parse_args()解析函数
    args = parser.parse_args()

    return args

# # 求解攻击优化时设置的epochs
# flags.DEFINE_integer('attack_epochs', 500, 'Number of epochs to use when solving the attack optimization')
# # 供tensorflow使用的随机种子
# flags.DEFINE_integer('tf_seed', 12345, 'The random seed for tensorflow to use')
# # 保存频率
# flags.DEFINE_integer("save_frequency", 50, "Save at every x-th epoch where x is the number specified at this parameter")


# # 是否进行裁剪 clip噪声的最大值 最小值 image+noise的最大值 最小值
# flags.DEFINE_boolean('clipping', True, 'Specifies whether to use clipping')
# flags.DEFINE_float('noise_clip_max', 0.5, 'The maximum value for clipping the noise, if clipping is set to True')
# flags.DEFINE_float('noise_clip_min', -0.5, 'The minimum value for clipping the noise, if clipping is set to True')
# flags.DEFINE_float('noisy_input_clip_max', 1.0,
#                    'The maximum value for clipping the image+noise, if clipping is set to True')
# flags.DEFINE_float('noisy_input_clip_min', 0.0,
#                    'The minimum value for clipping the image+noise, if clipping is set to True')
#
# # 指定是否使用inverse mask（将原始图像中的所有像素设置为指定的“值）
# flags.DEFINE_boolean('inverse_mask', True,
#                      'Specifies whether to use an inverse mask (set all pixels in the original image to a specified '
#                      'value)')
# # 使用反向时将原始图像中遮罩区域内的像素设置为的值蒙版
# flags.DEFINE_float('inverse_mask_setpoint', 0.5,
#                    'The value to set the pixels within the mask region in the original image to if using an inverse '
#                    'mask')
#
# flags.DEFINE_string('checkpoint', 'attack_single', 'Prefix to use when saving the checkpoint')
#
# # 使用300*300并将其裁剪成32*32,还是仅仅使用32*32  默认为False
# flags.DEFINE_boolean('fullres_input', False,
#                      'Specifies whether to use 300 by 300 input images (and resize them down to 32 by 32 for model '
#                      'input) or just use 32 by 32')
#
# # 存储输出图像的路径 普通输出路径和big image的输出路径
# flags.DEFINE_string('output_path', '', 'Filepath to where to save the image')
# flags.DEFINE_string('big_image', '', 'Filepath to big image')
#
#
# # flags.DEFINE_float('adam_learning_rate', 0.5, 'The value to set the pixels within the mask region in the original
# # image to')
