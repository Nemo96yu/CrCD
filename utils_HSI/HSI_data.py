import torch
import torch.utils.data as data
import spectral as spy
from utils.randaugment import RandomAugment
import torchvision.transforms as transforms
from utils.utils_algo import generate_uniform_cv_candidate_labels

class HSI_data(data.Dataset):
    '''
    引用时：train_data = fun_data(data_sets, cfg_data['train_data'])
    input：data_sets(data_sample)# dict_keys(['train_indices', 'train_num', 'test_indices', 'test_num','no_gt_indices', 'no_gt_num', 'pad_img', 'pad_img_indices', 'img_gt', 'ori_gt'])

    '''
    def __init__(self, data_sample, cfg, train_test, partial_rate):

        self.phase = cfg['phase']
        # img：pad_image
        # img_indices：每个patch的坐标集合
        self.img = data_sample['pad_img']  # torch.Size([103, 636, 366])
        self.img_indices = data_sample['pad_img_indices']   # torch.Size([207400, 4])
        self.gt = data_sample['img_gt']
        self.partialY = data_sample['partialY']
        self.pca = 'img_pca_pad' in data_sample
        self.tt = train_test
        self.partial_rate = partial_rate
        if self.pca:
            self.img_pca = data_sample['img_pca_pad']
        # data_indices：用img_gt的标签信息划分得到的样本
        if self.phase == 'train':
            self.data_indices = data_sample['train_indices']
        elif self.phase == 'test':
            self.data_indices = data_sample['test_indices']
        elif self.phase == 'no_gt':
            self.data_indices = data_sample['no_gt_indices']
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=21, scale=(0.2, 1.)),  # 32
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=21, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        # 该方法支持从 0 到 len(self)的索引
        # data_indices=: torch.Size([270, 3])    img_indices=: torch.Size([207400, 4])
        index = self.data_indices[idx]      # torch.Size([3])
        img_index = self.img_indices[index[0]]  # img_index 坐标
        partialY = self.partialY
        # 从pad_img中依据坐标截取样本
        # 为什么不截取好再直接读入呢？

        img = self.img[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]  # torch.Size([103, 27, 27])
        label = self.gt[index[1], index[2]]
        l = index[0]
        label_p = partialY[l, :]
        img_pca = self.img_pca[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        # img_w = self.weak_transform(img_pca)
        # img_s = self.strong_transform(img_pca)
        img_w = img
        img_s = img_pca

        return img_s, label_p, label, index, img_w, img_pca





def batch_collate(batch):
    # 用来处理不同情况下的输入dataset的封装
    images_s = []
    labels = []
    labels_p = []
    indices = []
    images_w = []
    images_pca = []
    for sample in batch:
        images_s.append(sample[0])
        labels_p.append(sample[1])
        labels.append(sample[2])
        indices.append(sample[3])
        images_w.append(sample[4])
        images_pca.append(sample[5])

    return torch.stack(images_w, 0), torch.stack(images_s, 0), torch.stack(labels_p), torch.stack(labels), torch.stack(indices),torch.stack(images_pca, 0)



