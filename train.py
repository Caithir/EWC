import time
import torch
from configs.config import config
from Utils import AverageMeter, accuracy
from Utils.logger import logger
from Utils import clip_and_track_grad


def train(train_loaders, model, criterion, optimizer, epoch, scheduled_actions=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batches in enumerate(zip(*train_loaders)):
        for b_id, (input, target) in enumerate(batches):
            # measure data loading time
            n_iter = (epoch * len(batches)) + i
            data_time.update(time.time() - end)

            if config.gpu is not None:
                input = input.cuda(config.gpu, non_blocking=True)
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss_input = [output, target]
            if config.experiments[0] == 'fisher':
                loss_input.append(model)
            loss = criterion(*loss_input)
            if config.experiments[0] == 'fisher':
                criterion.swap_task()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            if b_id == 1:
                logger.val_batch_log(prec1, loss, fisher=True)
                continue
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            clip_and_track_grad(model, config)
            # clip_grad_value_(model.parameters(), config.grad_clip)
            optimizer.step()



            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            logger.train_batch_log(model, top1.avg, losses.avg)
            if i % 5 ==0:
                logger.log_embedding(output, target, input, global_step=n_iter)
            if i % config.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loaders[0]), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
            if scheduled_actions:
                next(scheduled_actions)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if config.gpu is not None:
                input = input.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss_input = [output, target]

            if config.experiments[0] == 'fisher':
                loss_input.append(model)
            loss = criterion(*loss_input)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            logger.val_batch_log(top1.avg, losses.avg)
            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
