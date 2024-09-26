import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     header = f"Epoch: [{epoch}]"

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=warmup_factor, total_iters=warmup_iters
#         )

#     for images, targets in metric_logger.log_every(data_loader, print_freq, header):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

#         loss_value = losses_reduced.item()
#         print(losses, loss_value)

#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping training")
#             print(loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(losses).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             losses.backward()
#             optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     return metric_logger

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("step_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("epoch_loss", utils.SmoothedValue(window_size=len(data_loader), fmt="{avg:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 1:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # delete a background image (no boxes)
        images_list = []
        targets_list = []
        for i in range(len(images)):
#             print(targets[i]['boxes'].shape)
            if(targets[i]['boxes'].shape[1] == 4):
                images_list.append(images[i])
                targets_list.append(targets[i])
        images = tuple(images_list)
        targets = tuple(targets_list)
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
#         print('real_loss',losses.item())
#         print('reduced_loss', losses_reduced.item())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(step_loss=losses.item())
        metric_logger.update(epoch_loss=losses.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def compute_loss(model, data_loader, device):
    pass

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def post_processing(outputs, confident_threshold, area_threshold):
    if (confident_threshold > 0.0 and confident_threshold <= 1.0) or area_threshold > 0:
        cleaned_outputs = []
        for output in outputs:
            boxes = []
            labels = []
            scores = []
            # calculate a boundingbox area
            areas = (output['boxes'][:, 3] - output['boxes'][:, 1]) * (output['boxes'][:, 2] - output['boxes'][:, 0])

            for i in range(len(output['labels'])):
                if output['scores'][i].item() > confident_threshold and areas[i].item() > area_threshold:
                    boxes.append(output['boxes'][i].tolist())
                    labels.append(output['labels'][i].item())
                    scores.append(output['scores'][i].item())
            boxes = torch.tensor(boxes)
            labels = torch.tensor(labels)
            scores = torch.tensor(scores)

            cleaned_outputs.append({
                'boxes': boxes,
                'labels': labels,
                'scores': scores
            })
        return cleaned_outputs
    else:
        return outputs

@torch.inference_mode()
def evaluate(model, data_loader, device, confident_threshold=0.0, area_threshold=0):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        """
        
        ต้องเพิ่มการทำงาน lost ในนี้
        
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
#         print(outputs)
        
        # post processing
        outputs = post_processing(outputs, confident_threshold, area_threshold)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    
    coco_evaluator.summarize()
#     print(coco_evaluator)
#     mAP = coco_evaluator.coco_eval['bbox'].stats
#     print(f' AP@50:95 = {mAP[0]:.3f}')
#     print(f' AP@50    = {mAP[1]:.3f}')
#     print(f' AP@75    = {mAP[2]:.3f}')
    
    torch.set_num_threads(n_threads)
    return coco_evaluator
