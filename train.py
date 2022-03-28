import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from tensorboardX import SummaryWriter

from model.capl import PSPNet
from util import dataset 
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)
    np.random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed) 
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"        
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def get_new_proto(val_supp_loader, model, base_num=16, novel_num=5, init_gen_proto=False):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start New Proto Generation >>>>>>>>>>>>>>>>')

    model.eval()
    new_proto_num_epoch = 1#1
    with torch.no_grad():
        gened_proto_bed = torch.zeros(args.classes, 512).cuda()
        for epoch in range(new_proto_num_epoch):
            for i, (input, target, _, _) in enumerate(val_supp_loader):
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)                
                if main_process():
                    logger.info('Generating new prototypes {}/{}...'.format(epoch, new_proto_num_epoch))
                    logger.info('base_num: {}, novel_num: {}'.format(args.classes-args.novel_num, args.novel_num))
                    logger.info('Input: {}, Target: {}.'.format(input.shape, target.shape))

                input = input.contiguous().view(1, args.novel_num, args.shot, input.size(1), input.size(2), input.size(3))
                target = target.contiguous().view(1, args.novel_num, args.shot, target.size(1), target.size(2))
                input = input.repeat(8, 1, 1, 1, 1, 1)
                target = target.repeat(8, 1, 1, 1, 1)

                gened_proto = model(x=input, y=target, iter=i, eval_model=False, gen_proto=True, \
                        base_num=base_num, novel_num=novel_num)
                gened_proto = gened_proto.mean(0)
            gened_proto_bed = gened_proto_bed + gened_proto            
        gened_proto = gened_proto_bed / new_proto_num_epoch

    return gened_proto

def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    print('-------------------------Using Official BN--------------------------')
    BatchNorm = nn.BatchNorm2d
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PSPNet(layers=args.layers, classes=args.classes, \
        zoom_factor=args.zoom_factor, criterion=criterion, BatchNorm=BatchNorm, \
        pretrained=True, args=args)
    optimizer = torch.optim.SGD(
        [{'params': model.layer0.parameters()},
         {'params': model.layer1.parameters()},
         {'params': model.layer2.parameters()},
         {'params': model.layer3.parameters()},
         {'params': model.layer4.parameters()},
         {'params': model.ppm.parameters(), 'lr': args.base_lr * 10},
         {'params': model.cls.parameters(), 'lr': args.base_lr * 10},
         {'params': model.aux.parameters(), 'lr': args.base_lr * 10},
         {'params': model.main_proto, 'lr': args.base_lr * 10},
         {'params': model.aux_proto, 'lr': args.base_lr * 10},
         {'params': model.gamma_conv.parameters(), 'lr': args.base_lr * 10},
         ],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    
    if main_process():
        global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))          
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]



    val_transform = transform.Compose([
        transform.Resize(size=max(args.train_h, args.train_w)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, \
          transform=val_transform, shot=args.shot, seed=args.manual_seed, \
          data_split=args.data_split, use_coco=args.use_coco)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, worker_init_fn=worker_init_fn, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    val_supp_seed_list = args.val_supp_seed_list 
    val_supp_loader_list = []
    for val_supp_seed in val_supp_seed_list:
        print('processing val supp with seed: ',val_supp_seed)
        val_supp_data = dataset.SemData(split='val_supp', data_root=args.data_root, data_list=args.train_list, \
            transform=val_transform, shot=args.shot, seed=val_supp_seed, \
            data_split=args.data_split, use_coco=args.use_coco, val_shot=args.shot)      
        val_supp_loader = torch.utils.data.DataLoader(val_supp_data, worker_init_fn=worker_init_fn, batch_size=args.novel_num*args.shot, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)
        val_supp_loader_list.append(val_supp_loader)

    if args.only_evaluate: 
        mean_all_mIoU = 0
        mean_base_mIoU = 0
        mean_novel_mIoU = 0
        for val_supp_loader in val_supp_loader_list:
            gened_proto = get_new_proto(val_supp_loader, model, novel_num=val_supp_data.novel_class_num, base_num=val_supp_data.base_class_num, \
                                    init_gen_proto=False)
            loss_val, mIoU_val, mAcc_val, allAcc_val, base_mIoU, novel_mIoU = validate(val_supp_loader, val_loader, model, criterion, \
                                                novel_num=val_supp_data.novel_class_num, base_num=val_supp_data.base_class_num, \
                                                gened_proto=gened_proto.clone())     
            mean_all_mIoU += mIoU_val
            mean_base_mIoU += base_mIoU
            mean_novel_mIoU += novel_mIoU
        mIoU_val = mean_all_mIoU / len(val_supp_loader_list)
        base_mIoU = mean_base_mIoU / len(val_supp_loader_list)
        novel_mIoU = mean_novel_mIoU / len(val_supp_loader_list)
        print('Eval result: Final mIoU: {}, BASE: {}, NOVEL: {}'.format(mIoU_val, base_mIoU, novel_mIoU))
        exit(0)

    
    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_shot = 0
    if main_process():
        logger.info('Train shot: {}'.format(train_shot))
    train_data = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_list, \
        transform=train_transform, shot=train_shot, seed=args.manual_seed, \
        data_split=args.data_split,  use_coco=args.use_coco, val_shot=args.shot)

    test_shot = 1
    train_loader = torch.utils.data.DataLoader(train_data, worker_init_fn=worker_init_fn, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    max_iou = 0.
    filename = 'capl.pth'
    tmp_filename = 'capl.pth'       
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch) 
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)


        if args.evaluate and (epoch >= args.start_val_epoch or (epoch % 5 == 0 and not args.use_coco)): 
            mean_all_mIoU = 0
            mean_base_mIoU = 0
            mean_novel_mIoU = 0
            for val_supp_loader in val_supp_loader_list:
                gened_proto = get_new_proto(val_supp_loader, model, novel_num=val_supp_data.novel_class_num, base_num=val_supp_data.base_class_num, \
                                        init_gen_proto=False)
                loss_val, mIoU_val, mAcc_val, allAcc_val, base_mIoU, novel_mIoU = validate(val_supp_loader, val_loader, model, criterion, \
                                                    novel_num=val_supp_data.novel_class_num, base_num=val_supp_data.base_class_num, \
                                                    gened_proto=gened_proto.clone())     
                mean_all_mIoU += mIoU_val
                mean_base_mIoU += base_mIoU
                mean_novel_mIoU += novel_mIoU
            mIoU_val = mean_all_mIoU / len(val_supp_loader_list)
            base_mIoU = mean_base_mIoU / len(val_supp_loader_list)
            novel_mIoU = mean_novel_mIoU / len(val_supp_loader_list)
            print('Epoch: {}, Final mIoU: {}, BASE: {}, NOVEL: {}'.format(epoch, mIoU_val, base_mIoU, novel_mIoU))

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)


            if main_process():                
                if mIoU_val > max_iou and epoch_log > args.save_freq:
                    max_iou = mIoU_val
                    if os.path.exists(filename):
                        os.remove(filename)            
                    filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+ str(max_iou)+'_Base_'+str(base_mIoU)+'_Novel_'+str(novel_mIoU)+'.pth'
                    logger.info('Saving best checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
          
        # save the last epoch model  
        if main_process():                    
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename) 
            tmp_filename = args.save_path + '/tmp_' +str(epoch)+'.pth'     
            logger.info('Saving checkpoint to: ' + tmp_filename)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, tmp_filename)                


def train(train_loader, model, optimizer, epoch):
    torch.cuda.empty_cache()   
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power)

        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output, main_loss, aux_loss = model(x=input, y=target, iter=i)
        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        accuracy_meter.update(accuracy)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} ({main_loss_meter.avg:.4f}) '
                        'AuxLoss {aux_loss_meter.val:.4f} '                        
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f} ({accuracy_meter.avg:.4f}).'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy, accuracy_meter=accuracy_meter))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)



    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_supp_loader, val_loader, model, criterion, novel_num, base_num, gened_proto):
    torch.cuda.empty_cache() 
    if main_process():
        torch.cuda.empty_cache()  
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    gen_flag = 1

    with torch.no_grad():
        gened_proto = gened_proto.unsqueeze(0).repeat(8, 1, 1)        
        for i, (input, target, ori_size, ori_label) in enumerate(val_loader):
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(x=input, iter=i, eval_model=True, gen_proto=False, base_num=base_num, novel_num=novel_num, gened_proto=gened_proto)

            for b in range(input.size(0)):
                ## evaluate with the original labels where the padded ignored regions are set to 255. 
                tmp_output = output[b].unsqueeze(0)
                tmp_target = target[b].unsqueeze(0)
                tmp_ori_size = ori_size[b] # 2
                tmp_ori_label_1 = ori_label[b].unsqueeze(0)    # h, w
                tmp_ori_label = tmp_ori_label_1[:, :int(tmp_ori_size[0]), :int(tmp_ori_size[1])]

                longerside = int(max(tmp_ori_size[0], tmp_ori_size[1]))
                backmask = torch.ones(tmp_ori_label.shape[0], longerside, longerside).cuda()*255
                backmask[0, :int(tmp_ori_size[0]), :int(tmp_ori_size[1])] = tmp_ori_label
                tmp_target = backmask.clone().long()               

                tmp_output = F.interpolate(tmp_output, size=tmp_target.size()[1:], mode='bilinear', align_corners=True)

                tmp_output = tmp_output.max(1)[1]
                intersection, union, tmp_target = intersectionAndUnionGPU(tmp_output, tmp_target, args.classes, args.ignore_label)
                if args.multiprocessing_distributed:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(tmp_target)
                intersection, union, tmp_target = intersection.cpu().numpy(), union.cpu().numpy(), tmp_target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(tmp_target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            accuracy_meter.update(accuracy)
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % args.print_freq == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Accuracy {accuracy:.4f} ({accuracy_meter.avg:.4f}).'.format(i + 1, len(val_loader),
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              accuracy=accuracy, accuracy_meter=accuracy_meter))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    base_mIoU = np.mean(iou_class[:base_num])
    novel_mIoU = np.mean(iou_class[base_num:])
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        logger.info('Val mIoU--Base: {:.4f}, Novel: {:.4f}'.format(base_mIoU, novel_mIoU))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return 0., mIoU, mAcc, allAcc, base_mIoU, novel_mIoU

if __name__ == '__main__':
    main()
