import torch
import torch.nn as nn
from torchvision import transforms


def extract_pc(image, pc=3):
    channel, height, width = image.shape  # input float tensor image with CxHxW
    # data = image.view(channel, height*width)
    data = image.contiguous().view(channel, height * width)
    data_c = data - data.mean(dim=1).unsqueeze(1)
    u, s, vt = torch.svd(data_c.matmul(data_c.T))
    sorted_data, indices = s.sort(descending=True)

    image_pc = u[:, indices[0:pc]].T.matmul(data)

    return image_pc.view(pc, height, width)


def std_norm(image):  # input tensor image size with CxHxW
    image = image.permute(1, 2, 0).numpy()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.tensor(image).mean(dim=[0, 1]), torch.tensor(image).std(dim=[0, 1]))
    ])   # (x - mean(x))/std(x) normalize to mean: 0, std: 1

    return trans(image)


def one_zero_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    data = image.view(channel, height*width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = (data - data_min.unsqueeze(1))/(data_max.unsqueeze(1) - data_min.unsqueeze(1))
    # (x - min(x))/(max(x) - min(x))  normalize to (0, 1) for each channel

    return data.view(channel, height, width)


def pos_neg_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    data = image.view(channel, height*width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = -1 + 2 * (data - data_min.unsqueeze(1))/(data_max.unsqueeze(1) - data_min.unsqueeze(1))
    # -1 + 2 * (x - min(x))/(max(x) - min(x))  normalize to (-1, 1) for each channel

    return data.view(channel, height, width)


def construct_sample(image, window_size=27):
    '''
        function：construct sample,切分得到patch,储存每个patch的坐标值
        input: image：torch.size(103, 610, 340)
                window_size：27
        output：pad_image, batch_image_indices
    '''
    _, height, width = image.shape  # input float tensor image size with CxHxW

    half_window = int(window_size//2)  # 13
    # 使用输入边界的复制值来填充
    pad = nn.ReplicationPad2d(half_window)  # pad input NxCxHxW ReplicationPad2d((13, 13, 13, 13))
    pad_image = pad(image.unsqueeze(0)).squeeze(0)  # torch.Size([103, 636, 366])
    # print('pad_image.shape',pad_image.shape)

    # 用数组存储切分得到的patch的坐标
    batch_image_indices = torch.zeros((height*width, 4), dtype=torch.long)  # torch.Size([207400, 4])

    t = 0
    for h in range(height):
        for w in range(width):
            batch_image_indices[t, :] = torch.tensor([h, h + window_size, w, w + window_size])
            t += 1

    return pad_image, batch_image_indices


def label_transform(gt):
    '''
        function：tensor label to 0-n for training
        input: gt
        output：gt
        tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        -> tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    '''
    label = torch.unique(gt)   # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    gt_new = torch.zeros_like(gt)   # torch.size(610, 340)

    for each in range(len(label)):  # each 0~9
        indices = torch.where(gt == label[each])  # 2 tuple 两组索引数组来表示值的位置

        if label[0] == 0:
            gt_new[indices] = each - 1
        else:
            gt_new[indices] = each

    label_new = torch.unique(gt_new)  # tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])

    return gt_new


def label_inverse_transform(predict_result, gt):  # tensor result label to origin label
    label_origin = torch.unique(gt)
    label_predict = torch.unique(predict_result)

    predict_result_origin = torch.zeros_like(predict_result)
    for each in range(len(label_predict)):
        indices = torch.where(predict_result == label_predict[each])
        if len(label_predict) != len(label_origin):
            predict_result_origin[indices] = label_origin[each + 1]
        else:
            predict_result_origin[indices] = label_origin[each]

    return predict_result_origin


def select_sample(gt, ntr):  # input tensor data with NxCxHxW, tensor gt with HxW
    '''
        function: 用img_gt的标签信息划分样本
        input: gt -> torch.Size(610, 340)；  ntr -> train_set_num 30
        output：data_sample = {'train_indices': train_indices, 'train_num': train_num,
                   'test_indices': test_indices, 'test_num': test_num,
                   'no_gt_indices': no_gt_indices, 'no_gt_num': no_gt_num.unsqueeze(0) }
    '''
    gt_vector = gt.reshape(-1, 1).squeeze(1)  # torch.Size([207400])
    label = torch.unique(gt)   #torch.Size([10]) tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])

    first_time = True

    for each in range(len(label)):   # each 0~9
        indices_vector = torch.where(gt_vector == label[each])   # 1 tuple 返回一维的索引
        indices = torch.where(gt == label[each])  #  2 tuple 返回二维的索引

        indices_vector = indices_vector[0]
        indices_row = indices[0]
        indices_column = indices[1]

        # 背景 -1
        if label[each] == -1:
            no_gt_indices = torch.cat([indices_vector.unsqueeze(1),
                                       indices_row.unsqueeze(1),
                                       indices_column.unsqueeze(1)],
                                      dim=1
                                      )   # torch.Size([164624, 3])
            no_gt_num = torch.tensor(len(indices_vector))   # tensor(164624)

        # 其他标签 0-8
        else:
            class_num = torch.tensor(len(indices_vector))
            # each循环得到class_num：6631->18649->2099->3064->1345->5029->1330->3682->947

            # 得到sel_num    ntr = train_set_num 30
            if ntr < 1:   # 百分数
                ntr0 = int(ntr*class_num)

            else:
                ntr0 = ntr

            if ntr0 < 10:  # 控制sel_num的范围 10 ~ class_num//2
                sel_num = 10

            elif ntr0 > class_num//2:
                sel_num = class_num//2

            else:
                sel_num = ntr0

            sel_num = torch.tensor(sel_num)   # tensor(30)
            # 打乱
            rand_indices0 = torch.randperm(class_num)   # torch.randperm 给定参数n，返回一个从0到n-1的随机整数排列
            rand_indices = indices_vector[rand_indices0]  # torch.Size([6631])   indices_vector索引
            # 划分训练集train,测试集test
            # 划分打乱后的随机整数排列
            tr_ind0 = rand_indices0[0:sel_num]   # torch.Size([30])
            te_ind0 = rand_indices0[sel_num:]   # torch.Size([6601])   6601+30 = 6031
            # 划分随机整数排列对应的gt
            tr_ind = rand_indices[0:sel_num]  # torch.Size([30])
            te_ind = rand_indices[sel_num:]  # torch.Size([6601])
            # 训练集train：索引+坐标
            sel_tr_ind = torch.cat([tr_ind.unsqueeze(1),
                                    indices_row[tr_ind0].unsqueeze(1),
                                    indices_column[tr_ind0].unsqueeze(1)],
                                   dim=1
                                   )  # torch.Size([30, 3])
            # 测试集test
            sel_te_ind = torch.cat([te_ind.unsqueeze(1),
                                    indices_row[te_ind0].unsqueeze(1),
                                    indices_column[te_ind0].unsqueeze(1)],
                                   dim=1
                                   )  # torch.Size([6601, 3])

            if first_time:
                first_time = False

                train_indices = sel_tr_ind
                train_num = sel_num.unsqueeze(0)

                test_indices = sel_te_ind
                test_num = (class_num - sel_num).unsqueeze(0)

            else:  # [vector_indices, row_indices, column_indices] for train indices
                train_indices = torch.cat([train_indices, sel_tr_ind], dim=0)
                train_num = torch.cat([train_num, sel_num.unsqueeze(0)])
                # train_num tensor([30, 30])
                # train_num tensor([30, 30, 30])
                # ......
                # train_num tensor([30, 30, 30, 30, 30, 30, 30, 30, 30])
                # train_num.shape: torch.Size([2])--->torch.Size([9])

                test_indices = torch.cat([test_indices, sel_te_ind], dim=0)
                test_num = torch.cat([test_num, (class_num - sel_num).unsqueeze(0)])


    # 训练集
    rand_tr_ind = torch.randperm(train_num.sum())   # torch.Size([270])  = 30*9
    train_indices = train_indices[rand_tr_ind, ]   # torch.Size([270, 3])
    # 测试集
    rand_te_ind = torch.randperm(test_num.sum())   # torch.Size([42506])
    test_indices = test_indices[rand_te_ind, ]   # torch.Size([42506, 3])
    # 背景
    rand_no_gt_ind = torch.randperm(no_gt_num.sum())   # torch.Size([164624])
    no_gt_indices = no_gt_indices[rand_no_gt_ind, ]   # torch.Size([164624, 3])

    data_sample = {'train_indices': train_indices, 'train_num': train_num,
                   'test_indices': test_indices, 'test_num': test_num,
                   'no_gt_indices': no_gt_indices, 'no_gt_num': no_gt_num.unsqueeze(0)
                   }   #  list length 6

    return data_sample
