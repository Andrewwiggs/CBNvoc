from __future__ import division
import os
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNet
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.datasets import VOCDetection
import time
from collections import defaultdict
from torch.nn.parameter import Parameter
import torch.optim.lr_scheduler as lr_scheduler

def create_folder_structure():
    base_folder = 'CBN_None' # 保持一致，使用之前讨论的文件夹名
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    models = ['RetinaNet']
    norm_types = ['None', 'SyncBN', 'CBN']

    for model in models:
        model_folder = os.path.join(base_folder, model)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        for norm_type in norm_types:
            norm_folder = os.path.join(model_folder, norm_type)
            if not os.path.exists(norm_folder):
                os.makedirs(norm_folder)

def collate_fn(batch):
    return tuple(zip(*batch))

class CBNModule(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,
                 buffer_num=0, rho=1.0,
                 burnin=200, two_stage=True,#burnin设置
                 FROZEN=False, out_p=False):
        super(CBNModule, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.buffer_num = buffer_num
        self.max_buffer_num = buffer_num
        self.rho = rho
        self.burnin = burnin
        self.two_stage = two_stage
        self.FROZEN = FROZEN
        self.out_p = out_p

        self.iter_count = 0
        self.pre_mu = []
        self.pre_meanx2 = []
        self.pre_dmudw = []
        self.pre_dmeanx2dw = []
        self.pre_weight = []
        self.ones = torch.ones(self.num_features).cuda()

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def _update_buffer_num(self):
        if self.two_stage:
            if self.iter_count > self.burnin:
                self.buffer_num = self.max_buffer_num
            else:
                self.buffer_num = 0
        else:
            self.buffer_num = int(self.max_buffer_num * min(self.iter_count / self.burnin, 1.0))

    def forward(self, input, weight=None):
        self._check_input_dim(input)
        y = input.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(input.size(1), -1)

        if weight is None:
            weight = self.weight
            
        if self.training and self.burnin > 0:
            self.iter_count += 1
            self._update_buffer_num()

        if self.buffer_num > 0 and self.training and input.requires_grad:
            cur_mu = y.mean(dim=1)
            cur_meanx2 = torch.pow(y, 2).mean(dim=1)
            cur_sigma2 = y.var(dim=1)

            dmudw = torch.autograd.grad(cur_mu, weight, self.ones, retain_graph=True)[0]
            dmeanx2dw = torch.autograd.grad(cur_meanx2, weight, self.ones, retain_graph=True)[0]

            mu_all = torch.stack([cur_mu, ] + [tmp_mu + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for tmp_mu, tmp_d, tmp_w in zip(self.pre_mu, self.pre_dmudw, self.pre_weight)])
            meanx2_all = torch.stack([cur_meanx2, ] + [tmp_meanx2 + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for tmp_meanx2, tmp_d, tmp_w in zip(self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight)])
            sigma2_all = meanx2_all - torch.pow(mu_all, 2)

            re_mu_all = mu_all.clone()
            re_meanx2_all = meanx2_all.clone()
            re_mu_all[sigma2_all < 0] = 0
            re_meanx2_all[sigma2_all < 0] = 0
            count = (sigma2_all >= 0).sum(dim=0).float()
            mu = re_mu_all.sum(dim=0) / count
            sigma2 = re_meanx2_all.sum(dim=0) / count - torch.pow(mu, 2)

            self.pre_mu = [cur_mu.detach(), ] + self.pre_mu[:(self.buffer_num - 1)]
            self.pre_meanx2 = [cur_meanx2.detach(), ] + self.pre_meanx2[:(self.buffer_num - 1)]
            self.pre_dmudw = [dmudw.detach(), ] + self.pre_dmudw[:(self.buffer_num - 1)]
            self.pre_dmeanx2dw = [dmeanx2dw.detach(), ] + self.pre_dmeanx2dw[:(self.buffer_num - 1)]

            tmp_weight = torch.zeros_like(weight.data)
            tmp_weight.copy_(weight.data)
            self.pre_weight = [tmp_weight.detach(), ] + self.pre_weight[:(self.buffer_num - 1)]

        else:
            x = y
            mu = x.mean(dim=1)
            cur_mu = mu
            sigma2 = x.var(dim=1)
            cur_sigma2 = sigma2

        if not self.training or self.FROZEN:
            y = y - self.running_mean.view(-1, 1)
            if self.out_p:
                y = y / (self.running_var.view(-1, 1) + self.eps)**.5
            else:
                y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
            
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * cur_mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * cur_sigma2
            y = y - mu.view(-1, 1)
            if self.out_p:
                y = y / (sigma2.view(-1, 1) + self.eps)**.5
            else:
                y = y / (sigma2.view(-1, 1)**.5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'buffer={max_buffer_num}, burnin={burnin}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

def voc_eval(detections, ground_truths, class_id, ovthresh=0.5, use_07_metric=False):
    class_dets = []
    for img_id, dets in detections.items():
        for det in dets:
            if det['category_id'] == class_id:
                class_dets.append((img_id, det['score'], det['bbox']))
    
    class_dets.sort(key=lambda x: -x[1])
    
    npos = 0
    class_gts = defaultdict(list)
    for img_id, gts in ground_truths.items():
        for gt in gts:
            if gt['category_id'] == class_id:
                class_gts[img_id].append(gt)
                if not gt['difficult']:
                    npos += 1
    
    tp = np.zeros(len(class_dets))
    fp = np.zeros(len(class_dets))
    
    for i, (img_id, _, det_bbox) in enumerate(class_dets):
        if img_id not in class_gts:
            fp[i] = 1
            continue
            
        max_iou = -np.inf
        best_gt_idx = -1
        gts = class_gts[img_id]
        
        for j, gt in enumerate(gts):
            iou = bbox_iou(det_bbox, gt['bbox'])
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = j
        
        if max_iou >= ovthresh:
            if not gts[best_gt_idx]['used']:
                tp[i] = 1
                gts[best_gt_idx]['used'] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return rec, prec, ap

def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou
    
def get_retinanet_model(norm_type):
    num_classes = len(VOC_CLASSES) + 1
    if norm_type == 'SyncBN':
        model = retinanet_resnet50_fpn(weights=None, norm_layer=nn.SyncBatchNorm, num_classes=num_classes)
    elif norm_type == 'CBN':
        backbone = resnet_fpn_backbone('resnet50', weights=None,norm_layer=lambda num_features: CBNModule(num_features,
                                                                               buffer_num=3, # 缓存3个批次的统计量
                                                                               rho=1.0,      # 影响历史信息的权重
                                                                               burnin=200,   # 烧热期设置为 200 个迭代
                                                                               two_stage=True)) # 烧热期后立即使用上下文
        #backbone = resnet_fpn_backbone('resnet50', weights=None, norm_layer=CBNModule)
        model = RetinaNet(backbone=backbone, num_classes=num_classes)
    else:
        model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0, std=0.01)
            if module.bias is not None:
                if 'classification_head' in str(type(module)).lower():
                    prior_prob = 0.01
                    b = -np.log((1 - prior_prob) / prior_prob)
                    nn.init.constant_(module.bias, b)
                else:
                    nn.init.constant_(module.bias, 0)
    return model
    
# 修改 train_model 函数签名，增加 lr_log_file_path 参数
def train_model(model, train_loader, optimizer, scheduler, device, num_epochs, lr_log_file_path):
    model.train()
    loss_list = []
    
    print("\n检查模型参数是否可训练:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    
    print(f"\n初始学习率: {optimizer.param_groups[0]['lr']}")

    # 打开学习率日志文件，准备写入
    with open(lr_log_file_path, 'w') as lr_log_f:
        lr_log_f.write("Epoch,LearningRate\n") # 写入表头

        for epoch in range(num_epochs):
            if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
                
            running_loss = 0.0
            for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
                if len(images) == 0:
                    print("警告: 空batch跳过")
                    continue
                    
                images = list(image.to(device) for image in images)
                processed_targets = []
                
                for t in targets:
                    processed_targets.append({
                        'boxes': t['boxes'].to(device),
                        'labels': t['labels'].to(device)
                    })

                optimizer.zero_grad()
                loss_dict = model(images, processed_targets)
                
                if not loss_dict:
                    print("警告: 空loss_dict跳过")
                    continue
                    
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                
                running_loss += losses.item()
                
            loss_epoch = running_loss / len(train_loader)
            loss_list.append(loss_epoch)
            print(f"\nEpoch {epoch} 完成, 平均损失: {loss_epoch:.4f}")
            
            # 在每个epoch结束后更新学习率并记录
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr'] # 再次获取更新后的学习率
                print(f"Epoch {epoch} 后的学习率: {current_lr:.6f}")
            
            # 写入学习率到日志文件
            lr_log_f.write(f"{epoch},{current_lr:.10f}\n") # 保存到小数点后10位

    return loss_list

def evaluate_model(model, val_loader, device):
    model.eval()

    detections = defaultdict(list)
    ground_truths = defaultdict(list)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            if len(images) == 0:
                continue

            images = list(image.to(device) for image in images)
            outputs = model(images)

            for i, t in enumerate(targets):
                img_id = f"{batch_idx}_{i}"
                boxes = t['boxes'].cpu().numpy()
                labels = t['labels'].cpu().numpy()

                for box, label in zip(boxes, labels):
                    if 1 <= label <= len(VOC_CLASSES):
                        ground_truths[img_id].append({
                            'bbox': box.tolist(),
                            'category_id': int(label),
                            'difficult': False,
                            'used': False
                        })

            for i, output in enumerate(outputs):
                img_id = f"{batch_idx}_{i}"
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.01:
                        if 1 <= label <= len(VOC_CLASSES):
                            detections[img_id].append({
                                'bbox': box.tolist(),
                                'score': float(score),
                                'category_id': int(label)
                            })

    aps = []
    for cls_id in range(1, len(VOC_CLASSES) + 1):
        rec, prec, ap = voc_eval(detections, ground_truths, cls_id)
        aps.append(ap)

    mAP = np.mean(aps)
    end_time = time.time()

    total_images_processed = 0
    for images, _ in val_loader:
        total_images_processed += len(images)
    fps = total_images_processed / (end_time - start_time) if (end_time - start_time) > 0 else 0.0

    return mAP, fps


def get_voc_dataloader(batch_size, voc_root='VOC2007', distributed=False):
    if not os.path.exists(os.path.join(voc_root, 'VOCdevkit/VOC2007')):
        raise FileNotFoundError(f"VOC2007数据集目录结构不正确，请确保{voc_root}/VOCdevkit/VOC2007存在")

    def target_transform(target):
        boxes = []
        labels = []

        if isinstance(target, dict) and 'annotation' in target:
            for obj in target['annotation'].get('object', []):
                try:
                    bbox = obj['bndbox']
                    xmin = float(bbox['xmin'])
                    ymin = float(bbox['ymin'])
                    xmax = float(bbox['xmax'])
                    ymax = float(bbox['ymax'])

                    xmin = max(0.0, xmin)
                    ymin = max(0.0, ymin)
                    xmax = max(0.0, xmax)
                    ymax = max(0.0, ymax)

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])

                    cls_name = obj['name']
                    if cls_name in VOC_CLASSES:
                        cls_idx = VOC_CLASSES.index(cls_name) + 1
                        labels.append(cls_idx)
                    else:
                        labels.append(0)
                except (KeyError, ValueError) as e:
                    continue

        if not boxes:
            boxes = [[0., 0., 1., 1.]]
            labels = [0]

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        if labels_tensor.numel() > 0:
            labels_tensor = torch.clamp(labels_tensor, min=0, max=len(VOC_CLASSES))

        if boxes_tensor.numel() > 0:
            invalid_boxes_dim = (boxes_tensor[:, 2] <= boxes_tensor[:, 0]) | \
                                (boxes_tensor[:, 3] <= boxes_tensor[:, 1])
            if invalid_boxes_dim.any():
                valid_box_indices = ~invalid_boxes_dim
                boxes_tensor = boxes_tensor[valid_box_indices]
                labels_tensor = labels_tensor[valid_box_indices]
                if boxes_tensor.numel() == 0:
                    boxes_tensor = torch.as_tensor([[0., 0., 1., 1.]], dtype=torch.float32)
                    labels_tensor = torch.as_tensor([0], dtype=torch.int64)

        return {
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }
    
    img_transform = Compose([
        ToTensor(),
        torchvision.transforms.Resize((300, 300))
    ])

    train_dataset = VOCDetection(
        root=voc_root,
        year='2007',
        image_set='trainval',
        download=True,
        transform=img_transform,
        target_transform=target_transform
    )

    val_dataset = VOCDetection(
        root=voc_root,
        year='2007',
        image_set='test',
        download=True,
        transform=img_transform,
        target_transform=target_transform
    )
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        collate_fn=collate_fn
    )

    print(f"训练集样本数: {len(train_dataset)}")
    sample_idx = 0
    sample_img, sample_target = train_dataset[sample_idx]
    print(f"样本{sample_idx} - 图像shape: {sample_img.shape}")
    print("检查数据集中的样本格式:")
    for i in range(min(5, len(train_dataset))):
        _, sample_target = train_dataset[i]
        print(f"样本{i}的目标类型: {type(sample_target)}")
        
        if isinstance(sample_target, dict):
            print(f"样本{i}包含的键: {list(sample_target.keys())}")
            if 'boxes' in sample_target:
                print(f"目标框数量: {len(sample_target['boxes'])}")
            elif 'annotation' in sample_target:
                # This part is from the original target_transform logic, not the processed one.
                # It's better to check the processed target for consistency.
                pass
            else:
                print("警告: 无法识别的目标格式")
        else:
            print(f"警告: 不支持的目标类型: {type(sample_target)}")

    return train_loader, val_loader

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def main():
    if not os.path.exists('VOC2007/VOCdevkit/VOC2007'):
        print("请先下载VOC2007数据集并解压到VOC2007目录")
        return
    
    use_distributed = 'RANK' in os.environ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        print("Running in non-distributed mode")
    
    create_folder_structure()
    
    batch_sizes = [4]
    num_epochs = 100
    initial_learning_rate = 0.0005
    
    # scheduler_type = 'StepLR' #不同学习率调度器
    # step_size = 20
    # gamma = 0.1

    scheduler_type = 'CosineAnnealingLR'
    T_max = num_epochs
    eta_min = 1e-6

    # scheduler_type = None

    train_loader_initial_check, val_loader_initial_check = get_voc_dataloader(batch_sizes[0], distributed=use_distributed)
    print("训练样本数:", len(train_loader_initial_check.dataset))
    print("验证样本数:", len(val_loader_initial_check.dataset))

    models_info = {
        'RetinaNet': {
            'model_func': get_retinanet_model, 
            'optimizer': optim.Adam,
            'criterion': None
        },
    }

    norm_types = ['SyncBN','CBN','None']

    for batch_size in batch_sizes:
        train_loader, val_loader = get_voc_dataloader(batch_size, distributed=use_distributed)
        for model_name, info in models_info.items():
            for norm_type in norm_types:
                print(f"\n--- Starting training for Model: {model_name}, Norm: {norm_type}, Batch Size: {batch_size}, LR: {initial_learning_rate} ---")
                model = info['model_func'](norm_type).to(device)
                
                if use_distributed:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
                
                optimizer = info['optimizer'](model.parameters(), lr=initial_learning_rate)

                scheduler = None
                if scheduler_type == 'StepLR':
                    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                elif scheduler_type == 'CosineAnnealingLR':
                    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

                # 构建学习率日志文件的路径
                lr_log_file_path = os.path.join('CBN_None', model_name, norm_type,
                                                 f'lr_log_lr-{initial_learning_rate}_batch-{batch_size}_epoch-{num_epochs}.txt')

                loss_list = train_model(model, train_loader, optimizer, scheduler, device, num_epochs, lr_log_file_path) # 传递 lr_log_file_path
                loss_variance = np.var(loss_list)
                mAP, fps = evaluate_model(model, val_loader, device)

                scheduler_info_str = ""
                if scheduler_type == 'StepLR':
                    scheduler_info_str = f"_sch-step_ss-{step_size}_g-{gamma}"
                elif scheduler_type == 'CosineAnnealingLR':
                    scheduler_info_str = f"_sch-cosine_tm-{T_max}_emin-{eta_min}"


                result_file = os.path.join('CBN_None', model_name, norm_type,
                                           f'results_lr_{initial_learning_rate}_batch_{batch_size}_epoch_{num_epochs}{scheduler_info_str}.txt')
                with open(result_file, 'w') as f:
                    f.write(f'epochs: {num_epochs}\n')
                    f.write(f'Batch Size: {batch_size}\n')
                    f.write(f'Model: {model_name}\n')
                    f.write(f'Normalization Type: {norm_type}\n')
                    f.write(f'Initial Learning Rate: {initial_learning_rate}\n')
                    if scheduler_type:
                        f.write(f'Scheduler Type: {scheduler_type}\n')
                        if scheduler_type == 'StepLR':
                            f.write(f'Scheduler Params: Step Size={step_size}, Gamma={gamma}\n')
                        elif scheduler_type == 'CosineAnnealingLR':
                            f.write(f'Scheduler Params: T_max={T_max}, Eta_min={eta_min}\n')

                    f.write(f'mAP: {mAP}\n')
                    f.write(f'FPS: {fps}\n')
                    f.write(f'Loss Variance: {loss_variance}\n')
                print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()