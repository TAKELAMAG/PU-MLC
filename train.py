import os
import logging
import argparse
import math
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from utils.dataloader import get_data_loader, get_graph_and_word_file
from utils.checkpoint import load_pretrained_model
from src.loss_functions.vpu_loss import vpuLoss, mixup
from src.helper_functions.helper_functions import mAP, ModelEma
from src.helper_functions.metrics import AveragePrecisionMeter
from src.model.Backbone import Backbone
from src.model.Global_Branch import convert_to_lgconv


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='resnet01')
parser.add_argument('--model-path', type=str)
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--thre', default=0.8, type=float, metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int, metavar='N', help='print frequency (default: 64)')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dataset', default='COCO2014')
parser.add_argument('--prob', default=0.5, type=float)
parser.add_argument('--cropSize', default=448, type=int, metavar='N', help='input cropSize (default: 448)')
parser.add_argument('--scaleSize', default=512, type=int, metavar='N', help='input scaleSize (default: 448)')
parser.add_argument('--gamma', type=float, metavar='N', help='gamma')
parser.add_argument('--topK', default=1, type=int, metavar='N', help='topK (default:1)')
parser.add_argument('--pretrainedModel', type=str, help='pretrainedModelPath')
parser.add_argument('--alpha', type=float, metavar='N', help='alpha')
parser.add_argument('--ema', type=float, metavar='N', help='ema')
parser.add_argument('--Stop_epoch', type=int, metavar='N', help='Stop_epoch')


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def reduce_mean(tensor, nprocs):  
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

# free BN
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False
    
    # Bulid Logger
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = 'exp/log/{}.log'.format("train_"+str(args.prob))
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    # COCO Data loading
    train_loader, val_loader, sampler = get_data_loader(args)
    
    GraphFile, WordFile = get_graph_and_word_file(args, train_loader.dataset.changedLabels)

    # Setup model
    if local_rank == 0:
        print('creating model...')
    model = Backbone(GraphFile, WordFile, classNum=args.num_classes, topK=args.topK)
    if args.pretrainedModel != 'None':
        print("loading pretrained model...")
        model = load_pretrained_model(model, args)
    if local_rank == 0:
        print('done!')
    
    # Lgconv
    convert_to_lgconv(model.backbone)
    
    args.mix_alpha = 0.3
    
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True,
                                                        broadcast_buffers=False)
    
    weight_decay = 1e-4
    Epochs = 80
    criterion = {'vpuLoss': vpuLoss(gamma=args.gamma, alpha=args.alpha).cuda(),
                 'mixup': mixup(mix_alpha=args.mix_alpha).cuda()}
    
    for name, param in model.module.backbone.named_parameters():
        if "global_branch" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for param in model.module.backbone.layer4.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(filter(lambda param : param.requires_grad, model.module.parameters()), lr=args.lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args, sampler, local_rank, criterion, scheduler, steps_per_epoch, optimizer, Epochs)


def train_multi_label_coco(model, train_loader, val_loader, args, sampler, local_rank, criterion, scheduler, steps_per_epoch, optimizer, Epochs):
    ema = ModelEma(model, args.ema)
    
    # set optimizer
    Epochs = Epochs
    Stop_epoch = args.Stop_epoch

    highest_mAP = 0
    highest_OF1 = 0
    highest_CF1 = 0
    trainInfoList = []
    scaler = GradScaler()
    model.train()
    model.apply(fix_bn)
    for epoch in range(Epochs):
        sampler.set_epoch(epoch)
        if epoch > Stop_epoch:
            break
        for i, (sampleIndex, inputData, target, groundTruth) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()

            with autocast():
                output = model(input=inputData)  
            
            # vpu_loss
            loss_var = criterion['vpuLoss'](output, target)
            
            # mixup
            loss_reg = criterion['mixup'](model, inputData, target, output)
            
            loss = loss_var + loss_reg
            
            model.zero_grad()
            
            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()
            
            loss = reduce_mean(loss, torch.distributed.get_world_size())

            ema.update(model)
            # store information
            if i % 100 == 0 and local_rank == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.4f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        model.eval()
        mAP_score, OF1, CF1 = validate_multi(val_loader, model, ema, local_rank)
        model.train()
        model.apply(fix_bn)
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
        if OF1 > highest_OF1:
            highest_OF1 = OF1
        if CF1 > highest_CF1:
            highest_CF1 = CF1
        if local_rank == 0:
            print('c_mAP = {:.2f}, h_mAP = {:.2f}, h_OF1 = {:.3f}, h_CF1 = {:.3f}\n'
                 .format(mAP_score, highest_mAP, highest_OF1, highest_CF1))
            logger.info('c_mAP = {:.2f}, h_mAP = {:.2f}, h_OF1 = {:.3f}, h_CF1 = {:.3f}'
                 .format(mAP_score, highest_mAP, highest_OF1, highest_CF1))


def validate_multi(val_loader, model, ema_model, local_rank):
    if local_rank == 0:
        print("starting validation")
    Sig = torch.nn.Sigmoid()
    
    apMeter_regular = AveragePrecisionMeter()
    apMeter_ema = AveragePrecisionMeter()
    
    OF1_ret = 0
    CF1_ret = 0
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (sampleIndex, input, target, groundTruth) in enumerate(val_loader):
        target = target
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()
        
        target[target < 0] = 0
        apMeter_regular.add(output_regular.cpu().detach(), target.cpu().detach())
        apMeter_ema.add(output_ema.cpu().detach(), target.cpu().detach())

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    
    threshold = 0.01
    while(threshold < 1.0):
        OP_regular, OR_regular, OF1_regular, CP_regular, CR_regular, CF1_regular = apMeter_regular.overall(threshold)
        OP_ema, OR_ema, OF1_ema, CP_ema, CR_ema, CF1_ema = apMeter_ema.overall(threshold)

        if math.isnan(OF1_regular):
            OF1_regular = 0
        if math.isnan(OF1_ema):
            OF1_ema = 0
        if math.isnan(CF1_regular):
            CF1_regular = 0
        if math.isnan(CF1_ema):
            CF1_ema = 0
            
        recent_OF1 = max(OF1_regular, OF1_ema)
        recent_CF1 = max(CF1_regular, CF1_ema)
        if recent_OF1 > OF1_ret:
            OF1_ret = recent_OF1
        if recent_CF1 > CF1_ret:
            CF1_ret = recent_CF1
        threshold += 0.01
    
    if local_rank == 0:
        print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    mAP_ret = max(mAP_score_regular, mAP_score_ema)
    return mAP_ret, OF1_ret, CF1_ret


if __name__ == '__main__':
    main()
