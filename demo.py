
import argparse
import torchvision.models as models
from torchvision import transforms
import easydict as edict
from loader.data_loader import places365_loader
from util.common import *
from util.image_operations import *
from util.places365_categories import places365_categories

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--arch', default='resnet18_fc_ma', type=str, help='arch')
parser.add_argument('--dataset', default='places365', type=str, help='dataset')
parser.add_argument('--mark', default='t5', type=str, help='mark')
parser.add_argument('--nd_result', default='pretrained/tally.csv', type=str, help='netdissection result')
parser.add_argument('--modelfile', default='pretrained/resnet18_fc_ma_t5.pth', type=str, help='model file')


args = parser.parse_args()

settings = edict.EasyDict({
    "GPU" : True,
    "IMG_SIZE" : 224,
    "CNN_MODEL" : MODEL_DICT[args.arch],
    "DATASET" : args.dataset,
    "DATASET_PATH" : DATASET_PATH[args.dataset],
    "NUM_CLASSES" : NUM_CLASSES[args.dataset],
    "MODEL_FILE" : args.modelfile,
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
if not os.path.exists(os.path.join(settings.OUTPUT_FOLDER, 'vis')):
    os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'vis'))


def explain():

    def read_ndcsv(file):
        node_infos = {}
        with open(file, 'r') as f:
            for line in f.readlines()[1:]:
                infos = line.strip().split(',')
                node_infos[int(infos[0])] = (infos[2], float(infos[3]))
        return node_infos

    node_infos = read_ndcsv(args.nd_result)

    val_loader = places365_loader(settings, 'val', 1)
    model = settings.CNN_MODEL(num_classes=365)
    checkpoint = torch.load(settings.MODEL_FILE)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    mp1 = nn.Sequential(*list(model.children())[:8])
    preprocess = transforms.Compose([
                 transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(
                     mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]
                 )])


    # for c in range(365):
    #     ind_ = np.nonzero(model.rfc.weight[c] > 1e-3)
    #     wc, indc = model.rfc.weight[c, ind_].sort(0, True)
    #     inds = ind_[indc].squeeze()
    #
    #     print("{}\t".format(places365_categories[c]), end='')
    #     for ind in inds.numpy():
    #         print("{:.3f} ({} {} {:.2f})\t".format(model.rfc.weight[c, ind].item(), ind, node_infos[ind+1][0], node_infos[ind+1][1]), end='')
    #     print()


    lucky_dogs = [10012, 28993, 31316, 35436, 27791]
    for _ in range(10):
        lucky_dog = np.random.choice(len(val_loader.dataset.imgs))
        lucky_dogs.append(lucky_dog)
    infos = []

    for lucky_dog in lucky_dogs:
        with torch.no_grad():
            raw_img_path, target = val_loader.dataset.imgs[lucky_dog]
            org_img = PIL.Image.open(raw_img_path)
            img_tensor = preprocess(org_img)
            input_var = torch.autograd.Variable(img_tensor.unsqueeze(0))
            height, width = org_img.size
            org_img = np.array(org_img)
            fmap = mp1(input_var)

            scores = fmap.view(512, 49).mean(1) * model.rfc.weight[target]
            inds = scores.sort(0, True)[1][:5]
            percent = scores / scores.abs().sum() * 100

            info = []
            for ind in inds:
                ind = ind.item()
                info.append((ind + 1, node_infos[ind + 1][0].replace('-s',''), node_infos[ind + 1][1], percent[ind].item()))
                # print("{} {:.4f}:{:.2f}".format(node_infos[ind + 1][0], node_infos[ind + 1][1], scores[ind]))
            infos.append(info)

            percent_scale = percent.max().item()
            imgs = [imagalize(fmap[0, i].numpy()) for i in inds]
            vis_cams = []
            for ii, img in enumerate(imgs):
                vis_cam = ((info[ii][-1]/percent_scale) * 0.45 * cv2.applyColorMap(imresize(img, (width, height)), cv2.COLORMAP_JET)[:, :, ::-1] + org_img * 0.5)
                vis_cams.append(vis_cam)
            vis_arr = [PIL.Image.fromarray(org_img)] + [PIL.Image.fromarray(img.astype(np.uint8)) for img in vis_cams if
                                                        type(img) == np.ndarray]
            imsave(os.path.join(settings.OUTPUT_FOLDER, 'vis', 'cam_{}.jpg'.format(lucky_dog)),
                   imconcat(vis_arr, margin=2))

    from util.html import prefix, suffix, img_html
    html = [prefix]
    for lucky_dog, info in zip(lucky_dogs, infos):
        html.append(img_html(lucky_dog, places365_categories[lucky_dog // 100], info))
    html.append(suffix)
    with open(os.path.join(settings.OUTPUT_FOLDER, 'explain.html'), 'w') as f:
        f.write('\n'.join(html))


def val_resnet(model, val_loader):
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
    return top1, top5


def val():
    val_loader = places365_loader(settings, 'val')

    model = settings.CNN_MODEL(pretrained=settings.FINETUNE, num_classes=settings.NUM_CLASSES,
                               topk=int(args.mark.replace('t', '')))
    p(model)
    if settings.GPU:
        model = model.cuda()

    val_resnet(model, val_loader)

def main():
    # generate a html report(explain.html) located at [OUTPUT_FOLDER]
    explain()

    # give the validation score
    val()



if __name__ == '__main__':
    main()
