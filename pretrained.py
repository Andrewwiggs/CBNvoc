from __future__ import division

import os
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, ssd300_vgg16
#from yolov5.models.experimental import attempt_load
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
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

def create_folder_structure():
    base_folder = 'CBN_OneStage_Detection_Study'
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

def collate_fn(batch):#自定义的collate函数，用于处理数据集中不同尺寸的图像
    return tuple(zip(*batch))

def get_coco_dataloader(batch_size):#获取COCO数据集的加载器
    transform = Compose([ToTensor()])
    train_dataset = CocoDetection(root='coco2017/train2017', annFile='coco2017/annotations/instances_train2017.json',
                                  transform=transform)
    val_dataset = CocoDetection(root='coco2017/val2017', annFile='coco2017/annotations/instances_val2017.json',
                                transform=transform)

    # 添加分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            sampler=train_sampler, num_workers=4,
                            collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          sampler=val_sampler, num_workers=4,
                          collate_fn=collate_fn)
    return train_loader, val_loader

class CBNModule(nn.Module):#定义了一个名为CBNModule的PyTorch模块，用于实现CBN（Conditional Batch Normalization）
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,
                 buffer_num=0, rho=1.0,
                 burnin=0, two_stage=True,
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
        self.pre_meanx2 = []  # mean(x^2)
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

    def forward(self, input, weight=None):  # Make weight parameter optional
        self._check_input_dim(input)
        y = input.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(input.size(1), -1)

        # If weight is not provided, use self.weight (for compatibility with standard BN)
        if weight is None:
            weight = self.weight
            
        # burnin
        if self.training and self.burnin > 0:
            self.iter_count += 1
            self._update_buffer_num()

        if self.buffer_num > 0 and self.training and input.requires_grad:  # some layers are frozen!
            # cal current batch mu and sigma
            cur_mu = y.mean(dim=1)
            cur_meanx2 = torch.pow(y, 2).mean(dim=1)
            cur_sigma2 = y.var(dim=1)
            # cal dmu/dw dsigma2/dw
            dmudw = torch.autograd.grad(cur_mu, weight, self.ones, retain_graph=True)[0]
            dmeanx2dw = torch.autograd.grad(cur_meanx2, weight, self.ones, retain_graph=True)[0]
            # update cur_mu and cur_sigma2 with pres
            mu_all = torch.stack([cur_mu, ] + [tmp_mu + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for tmp_mu, tmp_d, tmp_w in zip(self.pre_mu, self.pre_dmudw, self.pre_weight)])
            meanx2_all = torch.stack([cur_meanx2, ] + [tmp_meanx2 + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for tmp_meanx2, tmp_d, tmp_w in zip(self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight)])
            sigma2_all = meanx2_all - torch.pow(mu_all, 2)

            # with considering count
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
            # TODO: outside **0.5?
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
            # TODO: outside **0.5?
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
    # 提取指定类别的检测结果
    class_dets = []
    for img_id, dets in detections.items():
        for det in dets:
            if det['category_id'] == class_id:
                class_dets.append((img_id, det['score'], det['bbox']))
    
    # 按置信度排序
    class_dets.sort(key=lambda x: -x[1])
    
    # 提取指定类别的真实标注
    npos = 0
    class_gts = defaultdict(list)
    for img_id, gts in ground_truths.items():
        for gt in gts:
            if gt['category_id'] == class_id:
                class_gts[img_id].append(gt)
                if not gt['difficult']:
                    npos += 1
    
    # 计算TP和FP
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
    
    # 计算precision-recall曲线
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    # 计算AP
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # 新式计算AP
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return rec, prec, ap

def bbox_iou(box1, box2):
    # 计算两个边界框的IoU
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
    # num_classes 应该包含背景类。VOC有20个类，所以如果模型将0作为背景，则num_classes=21
    num_classes = len(VOC_CLASSES) + 1 # 20个VOC类 + 1个背景类
    if norm_type == 'SyncBN':
        model = retinanet_resnet50_fpn(weights='DEFAULT', norm_layer=nn.SyncBatchNorm, num_classes=91) # 先用91类加载
    elif norm_type == 'CBN':
        # 如果要对骨干网络的BN层应用CBN，则需要用CBNModule构建backbone
        # 注意: weights='DEFAULT'只会加载卷积层的权重，CBNModule的参数会随机初始化
        backbone = resnet_fpn_backbone('resnet50', weights='DEFAULT', norm_layer=CBNModule)
        # 用这个自定义的骨干网络初始化RetinaNet，头部类别设置为COCO的91类
        model = RetinaNet(backbone=backbone, num_classes=91)
    else: # Default or 'None'
        model = retinanet_resnet50_fpn(weights='DEFAULT', num_classes=91) # 先用91类加载

    # 步骤2: 替换分类和边界框回归头部，使其适应你的VOC类别数量 (num_classes=21)
    # 获取头部输入的特征维度
    in_features_cls = model.head.classification_head.conv.in_channels
    in_features_reg = model.head.regression_head.conv.in_channels

    # 获取每个位置的锚框数量 (RetinaNet通常是9个)
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    # 根据 norm_type 创建新的头部
    # 注意：RetinaNetClassificationHead 和 RetinaNetRegressionHead 接受 norm_layer 参数
    head_norm_layer = None
    if norm_type == 'SyncBN':
        head_norm_layer = nn.SyncBatchNorm
    elif norm_type == 'CBN':
        head_norm_layer = CBNModule

    model.head.classification_head = RetinaNetClassificationHead(
        in_features_cls,
        num_anchors,
        num_classes, # 这里传入你期望的VOC类别数 (21)
        norm_layer=head_norm_layer
    )
    model.head.regression_head = RetinaNetRegressionHead(
        in_features_reg,
        num_anchors,
        norm_layer=head_norm_layer
    )

    # 新替换的头部默认是随机初始化的，通常不需要特别的初始化，
    # 但如果你想给分类头 bias 一个有利于早期收敛的值，可以这样：
    if hasattr(model.head.classification_head, 'conv'):
         prior_prob = 0.01
         b = -np.log((1 - prior_prob) / prior_prob)
         nn.init.constant_(model.head.classification_head.conv.bias, b)


    print(f"RetinaNet模型 (COCO预训练骨干网络，{num_classes}类头部) 使用 {norm_type} 规范化构建完成。")
    return model
    
    
def train_model(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    loss_list = []
    
    # 打印模型参数是否可训练
    print("\n检查模型参数是否可训练:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    
    # 打印优化器学习率
    print(f"\n初始学习率: {optimizer.param_groups[0]['lr']}")

    for epoch in range(num_epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
            
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
            # 检查数据变化
            if batch_idx < 3:  # 只打印前3个batch的统计信息
                print(f"\nBatch {batch_idx} 数据统计:")
                print(f"图像数量: {len(images)}")
                if len(images) > 0:
                    img_tensor = images[0]
                    print(f"图像形状: {img_tensor.shape}")
                    print(f"像素值范围: [{img_tensor.min().item():.3f}, {img_tensor.max().item():.3f}]")
                    print(f"像素均值: {img_tensor.mean().item():.3f}")
                
                for i, t in enumerate(targets[:min(1, len(targets))]):  # 只打印第一个目标的统计
                    print(f"目标{i}: boxes形状={t['boxes'].shape}, labels={t['labels'].unique()}")

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

            # 梯度清零检查
            optimizer.zero_grad()
            for param in model.parameters():
                if param.grad is not None:
                    print("警告: 在zero_grad()后仍有非None梯度!")
                    break

            loss_dict = model(images, processed_targets)
            
            if not loss_dict:
                print("警告: 空loss_dict跳过")
                continue
                
            losses = sum(loss for loss in loss_dict.values())
            
            # 打印详细损失信息
            print(f"\nEpoch {epoch} Batch {batch_idx} 损失详情:")
            for name, loss in loss_dict.items():
                print(f"{name}: {loss.item():.4f}")
            print(f"总损失: {losses.item():.4f}")

            # 反向传播检查
            losses.backward()
            
            # 检查梯度是否存在
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    break
            if not has_grad:
                print("警告: 反向传播后未检测到有效梯度!")
            else:
                # 打印部分梯度统计
                for name, param in list(model.named_parameters())[:3]:  # 只打印前3个参数
                    if param.grad is not None:
                        print(f"{name}梯度均值: {param.grad.mean().item():.6f}")

            # 参数更新检查
            optimizer.step()
            
            # 检查参数是否更新
            if batch_idx == 0 and epoch == 0:
                initial_params = {name: param.data.clone() for name, param in model.named_parameters()}
            elif batch_idx == 1 and epoch == 0:
                for name, param in model.named_parameters():
                    if not torch.equal(initial_params[name], param.data):
                        print(f"参数 {name} 已更新")
                    else:
                        print(f"警告: 参数 {name} 未更新!")

            running_loss += losses.item()
            
        loss_epoch = running_loss / len(train_loader)
        loss_list.append(loss_epoch)
        print(f"\nEpoch {epoch} 完成, 平均损失: {loss_epoch:.4f}")
        
    return loss_list
def evaluate_model(model, val_loader, device):
    model.eval()

    detections = defaultdict(list)
    ground_truths = defaultdict(list)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            if len(images) == 0:
                # print("警告: 空图像batch!")
                continue

            images = list(image.to(device) for image in images)
            outputs = model(images)

            # Process ground truths
            for i, t in enumerate(targets):
                img_id = f"{batch_idx}_{i}"
                boxes = t['boxes'].cpu().numpy()
                labels = t['labels'].cpu().numpy()

                for box, label in zip(boxes, labels):
                    # 在评估时，我们通常只对前景类别（1-20）感兴趣，跳过背景类（0）
                    if 1 <= label <= len(VOC_CLASSES):
                        ground_truths[img_id].append({
                            'bbox': box.tolist(),
                            'category_id': int(label), # 1-20
                            'difficult': False,
                            'used': False
                        })
                    # else:
                        # print(f"警告: 评估时发现背景真实标签 {label}，跳过。")


            # Get model predictions
            for i, output in enumerate(outputs):
                img_id = f"{batch_idx}_{i}"
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.01:
                        # 预测标签也需要是 1-20 范围内的前景类
                        if 1 <= label <= len(VOC_CLASSES): # 排除背景类 0
                            detections[img_id].append({
                                'bbox': box.tolist(),
                                'score': float(score),
                                'category_id': int(label) # 1-20
                            })
                        # else:
                            # print(f"警告: 评估时发现背景预测标签 {label}，跳过。")

    # Calculate VOC-style mAP
    aps = []
    # 迭代所有20个前景类别 (从 1 到 20)
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


def get_voc_dataloader(batch_size, voc_root='/root/autodl-tmp/VOC2007', distributed=False):
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

                    # 确保坐标有效且为正
                    xmin = max(0.0, xmin)
                    ymin = max(0.0, ymin)
                    xmax = max(0.0, xmax)
                    ymax = max(0.0, ymax)

                    # 验证边界框：xmax > xmin 和 ymax > ymin
                    if xmax <= xmin or ymax <= ymin:
                        # print(f"警告: 无效边界框 ([{xmin}, {ymin}, {xmax}, {ymax}])，跳过此对象。")
                        continue # 跳过无效边界框

                    boxes.append([xmin, ymin, xmax, ymax])

                    cls_name = obj['name']
                    if cls_name in VOC_CLASSES:
                        # 关键改变：标签从 1 到 len(VOC_CLASSES)，而不是 +1
                        # RetinaNet通常期望0是背景，1到num_classes-1是前景。
                        # 如果num_classes是21，则前景类是1-20。
                        cls_idx = VOC_CLASSES.index(cls_name) + 1 # 保持1-20的映射
                        labels.append(cls_idx)
                    else:
                        # print(f"警告: 未知类别 '{cls_name}'，使用背景类 (0)。")
                        labels.append(0) # 将未知类别映射到背景类 0
                except (KeyError, ValueError) as e:
                    # print(f"处理标注时出错: {e}，跳过此对象。")
                    continue

        # 处理图像中没有找到有效对象的情况
        if not boxes:
            # 提供一个虚拟的背景框，标签为背景类 0
            boxes = [[0., 0., 1., 1.]]
            labels = [0] # 明确设置为背景类 0

        # 转换为张量
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        # 最终检查和标签钳位（在上述逻辑下，应该是 0 到 len(VOC_CLASSES)）
        if labels_tensor.numel() > 0:
            # 确保标签在 0 到 num_classes-1 的范围内 (即 0 到 20)
            labels_tensor = torch.clamp(labels_tensor, min=0, max=len(VOC_CLASSES))

        # 检查处理后是否有边界框尺寸为零或负数
        if boxes_tensor.numel() > 0:
            invalid_boxes_dim = (boxes_tensor[:, 2] <= boxes_tensor[:, 0]) | \
                                (boxes_tensor[:, 3] <= boxes_tensor[:, 1])
            if invalid_boxes_dim.any():
                valid_box_indices = ~invalid_boxes_dim
                boxes_tensor = boxes_tensor[valid_box_indices]
                labels_tensor = labels_tensor[valid_box_indices]
                if boxes_tensor.numel() == 0: # 如果所有框都无效了
                    boxes_tensor = torch.as_tensor([[0., 0., 1., 1.]], dtype=torch.float32)
                    labels_tensor = torch.as_tensor([0], dtype=torch.int64) # 默认设为背景类 0

        return {
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }

    
    img_transform = Compose([
        ToTensor(),
        torchvision.transforms.Resize((300, 300))  # 统一调整尺寸
    ])


    train_dataset = VOCDetection(
        root=voc_root,
        year='2007',
        image_set='trainval',
        download=False,
        transform=img_transform,
    target_transform=target_transform # Use the refined target_transform
    )

    val_dataset = VOCDetection(
        root=voc_root,
        year='2007',
        image_set='test',
        download=False,
        transform=img_transform, # Also apply image transform to val_dataset for consistent input size
        target_transform=target_transform # Use the refined target_transform
    )
    # 修改采样器创建逻辑
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

    # 添加数据验证
    print(f"训练集样本数: {len(train_dataset)}")
    sample_idx = 0
    sample_img, sample_target = train_dataset[sample_idx]
    print(f"样本{sample_idx} - 图像shape: {sample_img.shape}")
    # 在get_voc_dataloader函数中添加
    # 修改数据验证部分
    print("检查数据集中的样本格式:")
    for i in range(min(5, len(train_dataset))):
        _, sample_target = train_dataset[i]
        print(f"样本{i}的目标类型: {type(sample_target)}")
        
        if isinstance(sample_target, dict):
            print(f"样本{i}包含的键: {list(sample_target.keys())}")
            if 'boxes' in sample_target:
                print(f"目标框数量: {len(sample_target['boxes'])}")
            elif 'annotation' in sample_target:
                print(f"原始标注中包含对象数量: {len(sample_target['annotation']['object'])}")
            else:
                print("警告: 无法识别的目标格式")
        else:
            print(f"警告: 不支持的目标类型: {type(sample_target)}")

    # 更健壮的目标框数量检查
    if isinstance(sample_target, dict):
        if 'annotation' in sample_target and 'object' in sample_target['annotation']:
            print(f"样本{sample_idx} - 目标框数量: {len(sample_target['annotation']['object'])}")
            if len(sample_target['annotation']['object']) > 0:
                print(f"样本{sample_idx} - 第一个目标框: {sample_target['annotation']['object'][0]}")
        else:
            print("警告: 样本中没有找到有效的annotation或object字段")
    else:
        print(f"警告: 样本目标格式不是dict, 实际类型: {type(sample_target)}")

    return train_loader, val_loader

# 添加VOC类别列表
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def main():
    # 在分布式初始化后添加数据集检查
    if not os.path.exists('VOC2007/VOCdevkit/VOC2007'):
        print("请先下载VOC2007数据集并解压到VOC2007目录")
        return
    
    # 初始化分布式训练
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
    num_epochs = 1
    #num_epochs = 10
    
    # 调试代码：检查数据加载
    train_loader, val_loader = get_voc_dataloader(batch_sizes[0], distributed=use_distributed)
    print("训练样本数:", len(train_loader.dataset))  # 应返回非零值
    print("验证样本数:", len(val_loader.dataset))    # 应返回非零值

    print("\n--- Inspecting first few training samples from DataLoader (After fix) ---")
    for i, (images, targets) in enumerate(train_loader):
        if i >= 5: # Inspect first 5 batches
            break
        print(f"Batch {i}: {len(images)} images")
        for j, img_targets in enumerate(targets):
            print(f"  Image {j}: Boxes shape={img_targets['boxes'].shape}, Labels shape={img_targets['labels'].shape}")
            if img_targets['boxes'].numel() > 0:
                print(f"    First box: {img_targets['boxes'][0].tolist()}")
            if img_targets['labels'].numel() > 0:
                unique_labels = torch.unique(img_targets['labels']).tolist()
                print(f"    Labels unique: {unique_labels}")
                if any(l < 0 or l > len(VOC_CLASSES) for l in unique_labels): # Check 0 to 20
                    print(f"!!! ERROR: Invalid labels found in DataLoader output: {unique_labels}")
                # 检查是否存在负数或大于num_classes-1的标签
                if (img_targets['labels'] < 0).any() or (img_targets['labels'] > len(VOC_CLASSES)).any():
                    print(f"!!! CRITICAL: Out-of-range labels detected: {img_targets['labels']}")
        if len(images) == 0:
            print(f"!!! Warning: Batch {i} is empty after collate_fn.")
    print("--- Finished inspecting training samples ---\n")
    models_info = {
        'RetinaNet': {
            'model_func': get_retinanet_model, 
            'optimizer': optim.Adam,
            'criterion': None  # RetinaNet内置损失函数
        },
    }
    # 定义norm_types列表
    norm_types = ['SyncBN','CBN','None']
    # 调试代码：检查数据加载
    train_loader, val_loader = get_voc_dataloader(batch_sizes[0], distributed=use_distributed)
    print("训练样本数:", len(train_loader.dataset))
    print("验证样本数:", len(val_loader.dataset))

    print("\n--- Inspecting first few training samples from DataLoader ---")
    for i, (images, targets) in enumerate(train_loader):
        if i >= 5: # Inspect first 5 batches
            break
        print(f"Batch {i}: {len(images)} images")
        for j, img_targets in enumerate(targets):
            print(f"  Image {j}: Boxes shape={img_targets['boxes'].shape}, Labels shape={img_targets['labels'].shape}")
            if img_targets['boxes'].numel() > 0:
                print(f"    First box: {img_targets['boxes'][0].tolist()}")
            if img_targets['labels'].numel() > 0:
                print(f"    Labels unique: {torch.unique(img_targets['labels']).tolist()}")
                if (img_targets['labels'] < 1).any() or (img_targets['labels'] > 20).any():
                    print(f"!!! Invalid labels found in DataLoader output: {img_targets['labels']}")
        if len(images) == 0:
            print(f"!!! Warning: Batch {i} is empty after collate_fn.")
    print("--- Finished inspecting training samples ---\n")
    
    models_info = {
        'RetinaNet': {
            'model_func': get_retinanet_model, 
            'optimizer': optim.Adam,
            'criterion': None
        },
    }

    for batch_size in batch_sizes:
        train_loader, val_loader = get_voc_dataloader(batch_size, distributed=use_distributed)
        for model_name, info in models_info.items():
            for norm_type in norm_types:  # 现在norm_types已定义
                model = info['model_func'](norm_type).to(device)
                
                if use_distributed:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
                
                optimizer = info['optimizer'](model.parameters(), lr=0.001)
                # 不再使用外部criterion
                loss_list = train_model(model, train_loader, optimizer, None, device, num_epochs)
                loss_variance = np.var(loss_list)
                mAP, fps = evaluate_model(model, val_loader, device)

                result_file = os.path.join('CBN_OneStage_Detection_Study', model_name, norm_type,
                                           f'results_batch_{batch_size}.txt')
                with open(result_file, 'w') as f:
                    f.write(f'Batch Size: {batch_size}\n')
                    f.write(f'Model: {model_name}\n')
                    f.write(f'Normalization Type: {norm_type}\n')
                    f.write(f'mAP: {mAP}\n')
                    f.write(f'FPS: {fps}\n')
                    f.write(f'Loss Variance: {loss_variance}\n')


if __name__ == "__main__":
    main()