import torch
import scipy.io as io
import utils_HSI.data_preprocess as pre_fun
from utils.utils_algo import generate_uniform_cv_candidate_labels


def get_train_test_set(cfg):
    '''
        function：划分数据集train,test
            加载数据集，转化为tensor，label transform，切分patch，储存每个patch的坐标值，由gt划分样本，最终得到data_sample
        input: cfg
        output：data_sample
            # dict_keys(['train_indices', 'train_num', 'test_indices', 'test_num',
            # 'no_gt_indices', 'no_gt_num', 'pad_img', 'pad_img_indices', 'img_gt', 'ori_gt'])
    '''

    # 从cfg导入设定好的参数
    data_path = cfg['data_path']   # '../datasets/PaviaU.mat'
    image_name = cfg['image_name']   # 'paviaU'
    gt_name = cfg['gt_name']   # 'paviaU_gt'
    train_set_num = cfg['train_set_num']   # 30
    patch_size = cfg['patch_size']   # 27

    # 数据加载 loadmat函数
    data = io.loadmat(data_path)

    img = data[image_name].astype('float32')  # .astype转换数组的数据类型  (610, 340, 103)
    gt = data[gt_name].astype('float32')  # 转换成float32  (610, 340)
    img = torch.from_numpy(img)  # 转tensor   # torch.Size(610, 340, 103)
    gt = torch.from_numpy(gt)  # torch.Size(610, 340)

    img = img.permute(2, 0, 1)   # 变换tensor的维度,把channel放到第一维CxHxW  # torch.Size(103, 610, 340)
    img = pre_fun.std_norm(img)  # 归一化 # torch.Size(103, 610, 340)
    # label transform  0~9 -> -1~8
    img_gt = pre_fun.label_transform(gt)  # torch.Size(610, 340) -> (610, 340)
    # construct_sample：切分patch，储存每个patch的坐标值
    img_pad, img_pad_indices = pre_fun.construct_sample(img, patch_size)  # torch.Size([103, 636, 366]), ([207400, 4])
    # select_sample：用img_gt的标签信息划分样本
    data_sample = pre_fun.select_sample(img_gt, train_set_num)
    # 得到的data_sample = {'train_indices': train_indices, 'train_num': train_num,
    #                    'test_indices': test_indices, 'test_num': test_num,
    #                    'no_gt_indices': no_gt_indices, 'no_gt_num': no_gt_num.unsqueeze(0)
    #                    }   #  list length 6

    # data_sample再添加几项内容
    data_sample['pad_img'] = img_pad
    data_sample['pad_img_indices'] = img_pad_indices
    data_sample['img_gt'] = img_gt
    data_sample['ori_gt'] = gt

    # 构造部分标签集
    labels = lables_list(img_gt)
    partialY = generate_uniform_cv_candidate_labels(labels, 0.1)
    data_sample['partialY'] = partialY
    # print('data_sample.keys()',data_sample.keys())
    # dict_keys(['train_indices', 'train_num', 'test_indices', 'test_num',
    # 'no_gt_indices', 'no_gt_num', 'pad_img', 'pad_img_indices', 'img_gt', 'ori_gt'])

    if cfg['pca'] > 0:
        img_pca = pre_fun.extract_pc(img, cfg['pca'])
        img_pca = pre_fun.one_zero_norm(img_pca)
        img_pca = pre_fun.std_norm(img_pca)

        img_pca_pad, _ = pre_fun.construct_sample(img_pca, patch_size)

        data_sample['img_pca_pad'] = img_pca_pad

    return data_sample, data_sample['partialY']

def lables_list(lables):
    t = 0
    label_true = []
    for i in range(610):
        for j in range(340):
            if lables[i][j] == -1:
                lables[i][j] = 0
            label_true.append(lables[i][j])
            t = t+1
    return label_true

