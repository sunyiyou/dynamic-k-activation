
import argparse

import torchvision.models as models
import easydict as edict
from util.common import *
from loader.data_loader import places365_loader

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--arch', default='resnet18_fc_ma', type=str, help='arch')
parser.add_argument('--dataset', default='places365', type=str, help='dataset')
parser.add_argument('--mark', default='t5', type=str, help='mark')

args = parser.parse_args()

settings = edict.EasyDict({
    "GPU" : True,
    "IMG_SIZE" : 224,
    "CNN_MODEL" : MODEL_DICT[args.arch],
    "DATASET" : args.dataset,
    "DATASET_PATH" : DATASET_PATH[args.dataset],
    "NUM_CLASSES" : NUM_CLASSES[args.dataset],
    "MODEL_FILE" : None,
    "FINETUNE": False,
    "WORKERS" : 12,
    "BATCH_SIZE" : 256,
    "PRINT_FEQ" : 10,
    "LR" : 0.1,
    "EPOCHS" : 90,
})

torch.manual_seed(0)

if settings.GPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

settings.OUTPUT_FOLDER = "result/pytorch_{}_{}_{}".format(args.arch, args.mark, args.dataset)
snapshot_dir = dir(os.path.join(settings.OUTPUT_FOLDER, 'snapshot'))
log_dir = dir(os.path.join(settings.OUTPUT_FOLDER, 'log'))
log_file = open(os.path.join(log_dir, "%s.txt" % time.strftime("%Y-%m-%d-%H:%M")), 'w')
print = log_f(log_file)


def train_resnet(model, train_loader, val_loader, dir=None):

    if settings.MODEL_FILE is not None:
        check_point = torch.load(settings.MODEL_FILE)
        state_dict = check_point['state_dict']
        model.load_state_dict(state_dict)
        epoch_cur = check_point['epoch']
    else:
        epoch_cur = -1

    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), settings.LR, momentum=0.9, weight_decay=1e-4)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = settings.LR * (0.1 ** (epoch // (settings.EPOCHS // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(epoch_cur+1, settings.EPOCHS):
        adjust_learning_rate(optimizer, epoch)

        print('Epoch[%d/%d]' % (epoch, settings.EPOCHS))
        # train
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        # if epoch == 2:
        #     p()
        for i, (input, target) in enumerate(train_loader):

            # if i > 21:
            #     break
            # input = torch.FloatTensor(input)
            data_time.update(time.time() - end)

            target = target.to(device=device)
            input = input.to(device=device)

            # measure data loading time

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            if loss > 10:
                p('loss explosion!!')
                break

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % settings.PRINT_FEQ == 0:
                print('Train: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    top1=top1, top5=top5))
        print(' * TIMESTAMP {}'.format(time.strftime("%Y-%m-%d-%H:%M")))
        print(' * TRAIN Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        torch.save({
            'epoch': epoch,
            'best_prec1': top1.avg,
            'state_dict': model.state_dict(),
        }, os.path.join(dir, 'epoch_%d.pth' % epoch))

def val_resnet(model, val_loader, epoch):
    if settings.MODEL_FILE is not None:
        check_point = torch.load(settings.MODEL_FILE)
        # state_dict = {str.replace(k, 'module.', ''): v for k, v in check_point[
        #     'state_dict'].items()}
        state_dict = check_point['state_dict']
        model.load_state_dict(state_dict)
    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # val
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device=device)
        input = input.to(device=device)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            fc_output = model(input_var)

            loss = criterion(fc_output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(fc_output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        if i % settings.PRINT_FEQ == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1, top5=top5))

    print(' * VAL Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

def main():
    val_loader = places365_loader(settings, 'val')
    train_loader = places365_loader(settings, 'train', shuffle=True, data_augment=True)

    # model = finetune_model
    model = settings.CNN_MODEL(pretrained=settings.FINETUNE, num_classes=settings.NUM_CLASSES,
                               topk=int(args.mark.replace('t', '')))
    p(model)

    if settings.GPU:
        model = nn.DataParallel(model).cuda()
        # model = model.cuda()
    model.train()

    train_resnet(model, train_loader, val_loader, snapshot_dir)
    for epoch in range(settings.EPOCHS - 10, settings.EPOCHS):
        settings.MODEL_FILE = "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format(args.arch, args.mark, args.dataset, epoch)
        val_resnet(model, val_loader, epoch)



if __name__ == '__main__':
    main()
