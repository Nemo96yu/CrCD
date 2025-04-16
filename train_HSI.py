import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from networkA import *
from networkB import *
from resnet import *
from utils.utils_algo import *
from utils.utils_loss import *
from utils_HSI import HSIdataloader
import network1
import network2
import scipy.io as io
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
torch.set_printoptions(precision=2, sci_mode=False)
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

parser.add_argument('--exp-dir', default='experiment/CrCD', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('--epochs', default=800, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')     # 256原
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,       # 原0.02
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--seed', default=6, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=6, type=int,
                    help='GPU id to use.')
parser.add_argument('--num-class', default=9, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=256, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=4096, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=80, type=int,
                    help='Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.1, type=float, 
                    help='ambiguity level (q)')

def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    model_path = './CrCD'
    args.exp_dir = model_path
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    print("=> creating model HSI_CNN")
    setup_seed(6)
    model1 = networkA(args, network1.SSMLP)
    model2 = networkB(args, network2.SSMLP)
    model1.cuda(6)
    model2.cuda(6)
    optimizer1 = torch.optim.SGD(model1.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer2 = torch.optim.SGD(model2.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model1.load_state_dict(checkpoint['state_dict1'])
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            model2.load_state_dict(checkpoint['state_dict2'])
            optimizer2.load_state_dict(checkpoint['optimizer2'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # load_data
    train_loader, partialY, train_sampler, test_loader = HSIdataloader.load_Indian(partial_rate=args.partial_rate, batch_size=args.batch_size)
    # this train loader is the partial label training loader
    print('Calculating uniform targets...')
    # calculate confidence
    tempY = partialY.sum(dim=1).unsqueeze(1).repeat(1, partialY.shape[1])
    confidence = partialY.float() / tempY
    confidence = confidence.cuda()
    # 损失函数
    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()
    loss_sem = Pro_MI_Loss()
    loss_stru = SinkhornDistance(0.5, 100)
    logger = None
    print('\nStart Training\n')

    best_acc = 0
    mmc = 0 #mean max confidence
    for epoch in range(args.start_epoch, 200):      # 200
        is_best = False
        start_upd_prot = epoch >= args.prot_start
        
        adjust_learning_rate(args, optimizer1, epoch)
        train(train_loader, model1, model2, loss_fn, loss_stru, loss_sem, loss_cont_fn, optimizer1, optimizer2, epoch, args, logger, start_upd_prot)
        loss_fn.set_conf_ema_m(epoch, args)

        acc_test, y_pred_test, y_test= test(model1, model2, test_loader, args, epoch, logger)
        pa, ua = paua(y_pred_test, y_test)
        mmc = loss_fn.confidence1.max(dim=1)[0].mean()
        
        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Acc {}, Best Acc {}. (lr {}, MMC {})\n'.format(epoch
                , acc_test, best_acc, optimizer1.param_groups[0]['lr'], mmc))
        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict1': model1.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'state_dict2': model2.state_dict(),
                'optimizer2': optimizer2.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))

def paua(predicted, gt):
    # 计算每个类别的像素总数
    class_counts = np.bincount(gt.astype('int64'), minlength=9)

    # 计算每个类别中正确分类的像素数量
    true_positive = np.bincount(predicted[predicted == gt], minlength=9)

    # 计算每个类别的生产者精度（PA）和用户精度（UA）
    pa = true_positive / class_counts
    # ua = true_positive / np.bincount(predicted, minlength=9)
    ua = cohen_kappa_score(gt, predicted)
    return pa, ua

def train(train_loader, model1, model2, loss_fn, loss_stru, loss_sem, loss_cont_fn, optimizer1, optimizer2, epoch, args,tb_logger, start_upd_prot=False):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc1@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc2@cls', ':2.2f')
    loss_cls1_log = AverageMeter('Loss@Cls1', ':2.2f')
    loss_cls2_log = AverageMeter('Loss@Cls2', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    loss_cont_p_log = AverageMeter('Loss@Cont_P', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls1_log, loss_cls2_log, loss_cont_log, loss_cont_p_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model1.train()
    model2.train()
    
    end = time.time()
    for i, (images_w, images_s, labels, true_labels, index, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        X_w, X_s, Y, index = images_w.cuda(6), images_s.cuda(6), labels, index.cuda(6)
        Y_true = true_labels.long().detach().cuda(6)
        # for showing training accuracy and will not be used when training
        cls_out2, key, score_prot2, Pro_dis2, prot2, W2 = model2(X_s, Y, args)
        cls_out1, q, score_prot1, Pro_dis1, prot1, W1 = model1(X_w, key, Y, args)

        batch_size = cls_out1.shape[0]

        # pseudo_target_cont = pseudo_target_cont1.contiguous().view(-1, 1)


        w = 1
        # 交叉消歧
        if start_upd_prot:
            w = 1
            loss_fn.confidence_update(temp_un_conf=score_prot1, batch_index=index, batchY=Y, rank=2)
            loss_fn.confidence_update(temp_un_conf=score_prot2, batch_index=index, batchY=Y, rank=1)
            # warm up ended

        # classification loss  分类损失
        loss_cls1 = loss_fn(cls_out1, index, 1)
        loss_cls2 = loss_fn(cls_out2, index, 2)
        loss_cont_stru = loss_stru(F.softmax(cls_out1, dim=1), F.softmax(cls_out2, dim=1))
        loss_cont = loss_cont_fn(q, mask=None, batch_size=batch_size)
        loss1 = loss_cls1 + args.loss_weight * loss_cont + w * loss_cont_stru
        loss2 = loss_cls2 + 0 * loss_cont

        loss_cls1_log.update(loss_cls1.item())
        loss_cls2_log.update(loss_cls2.item())
        loss_cont_log.update(loss_cont.item())
        loss_cont_p_log.update(loss_cont_stru.item())

        # log accuracy
        acc1 = accuracy(cls_out1, Y_true)[0]
        acc2 = accuracy(cls_out2, Y_true)[0]
        acc_cls.update(acc1[0])
        # acc = accuracy(score_prot, Y_true)[0]
        acc_proto.update(acc2[0])
        # compute gradient and do SGD step  retain_graph=True
        optimizer1.zero_grad()
        loss1.backward(retain_graph=True)
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    if args.gpu == 0:
        tb_logger.log_value('Train Acc1', acc_cls.avg, epoch)
        tb_logger.log_value('Train Acc2', acc_proto.avg, epoch)
        tb_logger.log_value('Classification Loss1', loss_cls1_log.avg, epoch)
        tb_logger.log_value('Classification Loss2', loss_cls2_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_p_log.avg, epoch)


def test(model1, model2, test_loader, args, epoch, tb_logger):
    with torch.no_grad():
        print('==> Evaluation...')
        model1.eval()
        model2.eval()
        top1_acc = AverageMeter("Top1", ':1.2f')
        top5_acc = AverageMeter("Top5", ':1.2f')
        count = 0
        y_pred_test = 0
        y_test = 0
        for batch_idx, (images_w, images_s, labels, true_labels, index, images_pca) in enumerate(test_loader):
            images_w, labels, images_s = images_w.cuda(6), true_labels.cuda(6), images_s.cuda(6)
            outputs1 = model1(images_w, args, eval_only=True)
            outputs2 = model2(images_s, args, eval_only=True)
            outputs = outputs1 * 0.2 + outputs2 * 0.8
            acc1, acc5 = accuracy(outputs, true_labels, topk=(1, 5))
            # if epoch == 186:
            for i in range(len(true_labels)):
                indexs = index[i]
                yy_test[indexs[1], indexs[2]] = np.argmax(outputs[i].cpu()) + 1
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])
            pre = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = pre
                y_test = true_labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, pre))
                y_test = np.concatenate((y_test, true_labels))
        acc_tensors = torch.Tensor([top1_acc.avg,top5_acc.avg]).cuda(args.gpu)
        
        print('Accuracy is %.2f%% (%.2f%%)'%(acc_tensors[0], acc_tensors[1]))
        if args.gpu == 0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('Top5 Acc', acc_tensors[1], epoch)             
    return acc_tensors[0], y_pred_test, y_test
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()
