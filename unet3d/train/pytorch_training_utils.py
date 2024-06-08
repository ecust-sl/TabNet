"""
Modified from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import shutil
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from scipy.special import softmax
from torch import Tensor
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
import numpy as np
try:
    from torch.utils.data._utils.collate import default_collate
except ModuleNotFoundError:
    # import from older versions of pytorch
    from torch.utils.data.dataloader import default_collate
from torch.nn.functional import cross_entropy as classify

def epoch_training(train_loader, model, criterion, metric,optimizer,optimizer_cls, epoch, n_gpus=None, print_frequency=1, regularized=False,
                   print_gpu_memory=False, vae=False, scaler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    #data_time = AverageMeter('Data', ':6.3f')
    seg_loss = AverageMeter('Loss_seg', ':.4e')
    losses = AverageMeter('Loss_all', ':.4e')
    cls_loss = AverageMeter('Loss_cls', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, seg_loss, cls_loss,losses],
        prefix="Epoch: [{}]".format(epoch))

    use_amp = scaler is not None

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target,tab_x,tab_y) in enumerate(train_loader):
        # measure data loading time
        #data_time.update(time.time() - end)

        if n_gpus:
            torch.cuda.empty_cache()
            if print_gpu_memory:
                for i_gpu in range(n_gpus):
                    print("Memory allocated (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.memory_allocated(i_gpu)))
                    print("Max memory allocated (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.max_memory_allocated(i_gpu)))
                    print("Memory cached (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.memory_cached(i_gpu)))
                    print("Max memory cached (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.max_memory_cached(i_gpu)))

        optimizer.zero_grad()
        optimizer_cls.zero_grad()
        #model.zero_grad()
        loss_seg, batch_size,loss_cls,cls_y,tab_y,acc = batch_loss(model, images, target, tab_x, tab_y, criterion=criterion, metric=metric, n_gpus=n_gpus, regularized=regularized,
                                      vae=vae, use_amp=use_amp)
        if n_gpus:
            torch.cuda.empty_cache()
        loss_all = loss_seg + loss_cls
        # measure accuracy and record loss
        losses.update(loss_all.item(), batch_size)
        seg_loss.update(loss_seg.item(), batch_size)
        cls_loss.update(loss_cls.item(), batch_size)
        #rocs.update(roc.item(), batch_size)

        if scaler:
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # compute gradient and do step
            loss_all.backward()
            optimizer.step()
            optimizer_cls.step()

        del loss_seg,loss_cls,loss_all

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_frequency == 0:
            progress.display(i)
    return losses.avg,cls_loss.avg


def batch_loss(model, images, target,tab_x,tab_y, criterion, metric,n_gpus=0, regularized=False, vae=False, use_amp=None):
    if n_gpus is not None:
        torch.cuda.empty_cache()
        images = images.cuda()
        target = target.cuda()
        tab_x = tab_x.cuda()
        tab_y = tab_y.cuda()
    # compute output
    if use_amp:
        from torch.cuda.amp import autocast
        with autocast():
            return _batch_loss(model, images, target,tab_x,tab_y, criterion,metric, regularized=regularized, vae=vae)
    else:
        return _batch_loss(model, images, target,tab_x,tab_y, criterion,metric, regularized=regularized, vae=vae)


def _batch_loss(model, images, target,tab_x,tab_y, criterion, metric,regularized=False, vae=False):
    output,cls_out,tab_x,tab_y = model(images,tab_x,tab_y)
    batch_size = images.size(0)
    if regularized:
        try:
            output, output_vae, mu, logvar = output
            loss = criterion(output, output_vae, mu, logvar, images, target)
        except ValueError:
            pred_y, pred_x = output
            loss = criterion(pred_y, pred_x, images, target)
    elif vae:
        pred_x, mu, logvar = output
        loss = criterion(pred_x, mu, logvar, target)
    else:
        loss = criterion(output, target)

    # cls_y = cls_out.cpu().detach().numpy()
    # cls_y = softmax(cls_y,axis=1)
    # sig = torch.nn.Sigmoid()
    # #sig = torch.nn.ReLU()
    # cls_out = sig(cls_out)
    loss2 = classify(cls_out, tab_y.long())
    tab_y = tab_y.cpu().detach().numpy()
    acc = accuracy_score(np.argmax(cls_out.cpu().detach().numpy(),axis=1),tab_y)
    #loss_all = loss + loss2
    #roc = metric(cls_y, cls_x)

    return loss, batch_size, loss2,cls_out,tab_y,torch.tensor(acc)


def epoch_validatation(val_loader, model, criterion, metric, n_gpus, print_freq=1, regularized=False, vae=False, use_amp=False):
    batch_time = AverageMeter('Time', ':6.3f')
    seg_loss = AverageMeter('Loss_seg', ':.4e')
    losses = AverageMeter('Loss_all', ':.4e')
    cls_loss = AverageMeter('Loss_cls', ':.4e')
    #rocs = AverageMeter('Roc', ':.4e')
    accs = AverageMeter('Acc', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses,accs],
        prefix='Validation: ')

    # switch to evaluate mode
    model.eval()
    cls_y_list = []
    tab_y_list = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target,tab_x,tab_y) in enumerate(val_loader):
            loss_seg, batch_size,loss_cls,cls_y,tab_y,acc = batch_loss(model, images, target,tab_x,tab_y, criterion, metric=metric,n_gpus=n_gpus, regularized=regularized,
                                          vae=vae, use_amp=use_amp)
            cls_y_list.append(cls_y)
            tab_y_list.append(tab_y)
            # measure accuracy and record loss
            loss_all = loss_seg + loss_cls
            # measure accuracy and record loss
            losses.update(loss_all.item(), batch_size)
            seg_loss.update(loss_seg.item(), batch_size)
            cls_loss.update(loss_cls.item(), batch_size)
            accs.update(acc.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

            if n_gpus:
                torch.cuda.empty_cache()
    #y_true, scores = stack_batches(tab_y_list, cls_y_list)
    #rocs = metric(y_true, scores)
    rocs = None
    return losses.avg,cls_loss.avg,accs.avg

def stack_batches(list_y_true, list_y_score):
    y_true = np.hstack(list_y_true)
    y_score = np.vstack(list_y_score)
    y_score = softmax(y_score, axis=1)
    return y_true, y_score

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def human_readable_size(size, decimal_places=1):
    for unit in ['', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def collate_flatten(batch, x_dim_flatten=5, y_dim_flatten=2):
    x, y = default_collate(batch)
    if len(x.shape) > x_dim_flatten:
        x = x.flatten(start_dim=0, end_dim=len(x.shape) - x_dim_flatten)
    if len(y.shape) > y_dim_flatten:
        y = y.flatten(start_dim=0, end_dim=len(y.shape) - y_dim_flatten)
    return [x, y]


def collate_5d_flatten(batch, dim_flatten=5):
    return collate_flatten(batch, x_dim_flatten=dim_flatten, y_dim_flatten=dim_flatten)
