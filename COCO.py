#加载COCO预训练权重，在终端中输入python COCO.py即可运行，若要使用分布式方法，需要在终端中输入以下命令：
# python -m torch.distributed.run --nproc_per_node=2 COCO.py
# 其中nproc_per_node为GPU数量。
from __future__ import division
import math
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim.lr_scheduler as lr_scheduler #引入学习率调度器
import torchvision
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor
from torchvision.models.detection import retinanet_resnet50_fpn,retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead, RetinaNet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim #引入优化器
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.v2 as T
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from tqdm import tqdm
import time
from collections import defaultdict
import datetime # 导入 datetime 模块
import json # 用于保存mAP和loss曲线数据
import matplotlib.pyplot as plt # 可用于绘图
import warnings # 用于警告




#1.创建文件夹结构和数据预处理
def create_folder_structure():# 创建文件夹结构，用于保存模型和结果
    base_folder = 'CBN_COCO'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    models = ['RetinaNet'] # 用以拓展其他模型 
    norm_types = ['None', 'SyncBN', 'CBN']
    for model in models: 
        model_folder = os.path.join(base_folder, model)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        for norm_type in norm_types:
            norm_folder = os.path.join(model_folder, norm_type)
            if not os.path.exists(norm_folder):
                os.makedirs(norm_folder)

def collate_fn(batch): # 处理已经转换好的 Tensor 数据且确保它能处理可能存在的空 boxes/labels
    """
    处理批次数据
    Args:
        batch: List[Tuple[Tensor, Dict]] - 批次数据列表，每个元素是(图像Tensor, 目标字典)
    Returns:
        Tuple[List[Tensor], List[Dict]] - 处理后的图像列表和目标列表
    """
    images, targets = tuple(zip(*batch))
    images = list(images)
    targets = list(targets)
    return images, targets

VOC_CLASSES = [ # 定义VOC类别，确保与get_voc_dataloader中的一致
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOC_Transforms:# 定义VOC数据转换类，用于处理图像和目标
    def __init__(self, train):
        self.raw_target_parser = self._create_raw_target_parser()
        # v2 transforms 负责图像和 bbox 的缩放、类型转换等
        transforms_list = [
            T.Resize((300, 300)),
            T.ToImage(), # 将 PIL Image 转换为 v2 Image 类型
            # 确保在 ToDtype 之前，数据是 v2 内部的 Image 类型，或者直接是 Tensor
            T.ToDtype(torch.float32, scale=True), # 转换为 float 张量，并缩放到 [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# 规范化
        ]
        self.v2_image_target_transforms = T.Compose(transforms_list)
    def _create_raw_target_parser(self):# 解析原始 XML 数据
        def parse_target(target):# 解析目标字典
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
                        xmax = max(xmin + 1.0, xmax) # 确保 xmax > xmin
                        ymax = max(ymin + 1.0, ymax) # 确保 ymax > ymin

                        boxes.append([xmin, ymin, xmax, ymax])

                        cls_name = obj['name']
                        if cls_name in VOC_CLASSES:
                            cls_idx = VOC_CLASSES.index(cls_name) + 1 # 类别ID从1开始，0留给背景
                            labels.append(cls_idx)
                        else:
                            labels.append(0) # 将未识别的类别映射为背景类
                    except (KeyError, ValueError) as e:
                        continue
            # 如果没有有效框，确保返回空的 Tensor，而不是让它变 None 或添加背景占位符
            if not boxes:
                boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
                labels_tensor = torch.empty((0,), dtype=torch.int64)
            else:
                boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

                # 确保标签在有效范围内 [0, num_classes] (0是背景，如果你的模型支持)
                if labels_tensor.numel() > 0:
                    labels_tensor = torch.clamp(labels_tensor, min=0, max=len(VOC_CLASSES))

                # 进一步过滤无效边界框
                if boxes_tensor.numel() > 0:
                    invalid_boxes_dim = (boxes_tensor[:, 2] <= boxes_tensor[:, 0]) | \
                                        (boxes_tensor[:, 3] <= boxes_tensor[:, 1])
                    if invalid_boxes_dim.any():
                        valid_box_indices = ~invalid_boxes_dim
                        boxes_tensor = boxes_tensor[valid_box_indices]
                        labels_tensor = labels_tensor[valid_box_indices]
                        # 再次检查过滤后是否变空
                        if boxes_tensor.numel() == 0:
                            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
                            labels_tensor = torch.empty((0,), dtype=torch.int64)
            return {
                'boxes': boxes_tensor,
                'labels': labels_tensor
            }
        return parse_target
    def __call__(self, img, raw_target_dict):
        # 1. 先用 raw_target_parser 解析原始 XML 字典，得到 Tensor 格式的目标
        processed_target = self.raw_target_parser(raw_target_dict)
        # 2. 然后将图像和 Tensor 格式的目标一起传入 v2 transforms 进行图像和 bbox 的同步转换
        # T.Compose 期望接收一个 (image, target) 元组
        return self.v2_image_target_transforms(img, processed_target)

#2.核心代码
class CBNModule(nn.Module):#本模型中使用的经修改的CBN
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
        # 将列表改为注册 buffer，以确保在 DDP 中正确同步和保存加载模型
        # 或者至少确保这些列表在 forward 内部被正确地处理为张量，并保持尺寸一致
        # 对于历史梯度和权重，需要确保它们是张量，并且在 .detach() 后存储
        # 鉴于它们是列表，需要手动管理它们的 device 和 dtype
        self.pre_mu = []
        self.pre_meanx2 = []  # mean(x^2)
        self.pre_dmudw = []
        self.pre_dmeanx2dw = []
        self.pre_weight = []
        self.ones = torch.ones(self.num_features) 

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
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            if self.running_mean is not None:
                self.running_mean.zero_()
            if self.running_var is not None:
                self.running_var.fill_(1)
        if self.affine:
            if self.weight is not None:
                self.weight.data.uniform_()
            if self.bias is not None:
                self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def _update_buffer_num(self):
        if self.two_stage:
            if self.iter_count > self.burnin:
                self.buffer_num = self.max_buffer_num
            else:
                self.buffer_num = 0
        else:
            # 避免除以零，如果 burnin 为 0，则直接设为 max_buffer_num
            if self.burnin == 0:
                 self.buffer_num = self.max_buffer_num
            else:
                 self.buffer_num = int(self.max_buffer_num * min(self.iter_count / self.burnin, 1.0))
    # 修改: forward 方法签名不再包含 weight 参数 
    def forward(self, input):
        self._check_input_dim(input)
        # 确保 self.ones 和 input 在同一个设备上
        if self.ones.device != input.device:
            self.ones = self.ones.to(input.device)
        y = input.transpose(0, 1) # 交换 batch 和 channel 维度，以便每个 channel 都被视为一个特征
        return_shape = input.shape 
        y = y.contiguous().view(input.size(1), -1) # input.size(1) 是 C (num_features)
        # burnin 逻辑
        if self.training and self.burnin > 0:# 训练阶段
            self.iter_count += 1
            self._update_buffer_num()
        # 初始化 mu 和 sigma2，以防后面的 if 分支不执行
        mu = None
        sigma2 = None
        if self.buffer_num > 0 and self.training and input.requires_grad:
            cur_mu = y.mean(dim=1)
            cur_meanx2 = torch.pow(y, 2).mean(dim=1)
            # 重点调试区域：泰勒展开项和梯度计算 
            # 确保 self.weight 是可导的，并且在正确的设备上
            if not self.affine or not self.weight.requires_grad:
                # 如果 weight 不可导，则 dmudw 和 dmeanx2dw 会是 None
                # 这会导致后续计算错误。通常 BN 的 weight 都是可导的。
                # 若网络部分被冻结，需要确保只对需要求导的层进行此操作。
                print("WARNING: CBNModule: self.weight is not affine or not requires_grad. Skipping grad calc for mu/meanx2.")
                dmudw = torch.zeros_like(self.weight)
                dmeanx2dw = torch.zeros_like(self.weight)
            else:
                try:
                    # 使用 input 而不是 y，因为 y 可能是 view 过的，可能导致 autograd 链断裂
                    # 或者确保 y 在求导时仍然连接到 input
                    # 确保 self.ones 也在同一个设备上
                    dmudw = torch.autograd.grad(cur_mu, self.weight, self.ones.to(self.weight.device), retain_graph=True)[0]
                    dmeanx2dw = torch.autograd.grad(cur_meanx2, self.weight, self.ones.to(self.weight.device), retain_graph=True)[0]
                    # 钳制梯度，防止其发散
                    # 如果 dmudw/dmeanx2dw 已经是 NaN/Inf，则无法钳制
                    if dmudw.isnan().any() or dmudw.isinf().any():
                        print(f"!!! DEBUG: dmudw is NaN/Inf at iter {self.iter_count}. Min/Max: {dmudw.min().item()}/{dmudw.max().item()}")
                        # 尝试将 NaN/Inf 梯度设置为 0.0，避免梯度爆炸
                        dmudw[dmudw.isnan()] = 0.0
                        dmudw[dmudw.isinf()] = 0.0
                    if dmeanx2dw.isnan().any() or dmeanx2dw.isinf().any():
                        print(f"!!! DEBUG: dmeanx2dw is NaN/Inf at iter {self.iter_count}. Min/Max: {dmeanx2dw.min().item()}/{dmeanx2dw.max().item()}")
                        dmeanx2dw[dmeanx2dw.isnan()] = 0.0
                        dmeanx2dw[dmeanx2dw.isinf()] = 0.0

                except RuntimeError as e:# 捕获所有可能的梯度计算错误
                    print(f"!!! WARNING: torch.autograd.grad failed: {e}. Setting dmudw/dmeanx2dw to zero.")
                    dmudw = torch.zeros_like(self.weight)
                    dmeanx2dw = torch.zeros_like(self.weight)
                    # 此时应该检查模型的计算图是否被正确构建

            # 确保历史缓冲区不为空且大小一致，防止索引错误
            if not (len(self.pre_mu) == len(self.pre_dmudw) == len(self.pre_weight)):
                self.pre_mu.clear()
                self.pre_meanx2.clear()
                self.pre_dmudw.clear()
                self.pre_dmeanx2dw.clear()
                self.pre_weight.clear()
                print("WARNING: History buffers inconsistent. Cleared all.")


            # 补偿历史统计量
            mu_all = [cur_mu]
            meanx2_all = [cur_meanx2]
            for tmp_mu, tmp_d_mu, tmp_meanx2, tmp_d_meanx2, tmp_w in zip(
                self.pre_mu, self.pre_dmudw, self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight
            ):
                # 确保所有操作在同一设备上
                tmp_d_mu_on_device = tmp_d_mu.to(self.weight.device)
                tmp_d_meanx2_on_device = tmp_d_meanx2.to(self.weight.device)
                tmp_w_on_device = tmp_w.to(self.weight.device)
                # 补偿项：(self.rho * tmp_d * (self.weight.data - tmp_w))
                # 检查 tmp_d 和 (self.weight.data - tmp_w) 的形状和操作
                # (self.rho * tmp_d * (self.weight.data - tmp_w)) 结果是 (num_features,)
                # sum(1).sum(1).sum(1) 在这里是错的，因为 dmudw 已经是 (num_features,)
                # 只有当 dmudw 是 (C, H, W) 这种形式时才需要 sum。
                # 修正为直接相乘
                delta_mu = self.rho * tmp_d_mu_on_device * (self.weight.data - tmp_w_on_device)
                delta_meanx2 = self.rho * tmp_d_meanx2_on_device * (self.weight.data - tmp_w_on_device)
                # 钳制 delta_mu 和 delta_meanx2，防止其过大导致溢出
                delta_mu = torch.clamp(delta_mu, min=-1e3, max=1e3) 
                delta_meanx2 = torch.clamp(delta_meanx2, min=-1e3, max=1e3) 
                # 确保 tmp_mu 和 tmp_meanx2 也在正确的设备上
                mu_all.append(tmp_mu.to(self.weight.device) + delta_mu)
                meanx2_all.append(tmp_meanx2.to(self.weight.device) + delta_meanx2)
            
            mu_all_tensor = torch.stack(mu_all)
            meanx2_all_tensor = torch.stack(meanx2_all)
            # 根据算法，应该在每个历史统计量上应用 max[nu, mu^2]
            # 计算 sigma2_all
            sigma2_all = meanx2_all_tensor - torch.pow(mu_all_tensor, 2)
            # **重要：钳制 sigma2_all，防止负值**
            sigma2_all = torch.clamp(sigma2_all, min=0.0) # 钳制为非负
            # 对应算法中的 max[nu, mu^2]
            # 这里是先计算 sigma2_all，然后根据 sigma2_all 是否小于 0 来清零
            # 聚合统计量
            count = (sigma2_all >= 0).sum(dim=0).float() # 计算有效统计量的数量
            # 防止除以零
            count = torch.max(count, torch.tensor(1.0, device=count.device)) # 至少为 1
            mu = mu_all_tensor.sum(dim=0) / count
            # 聚合后的 meanx2 也要确保不小于 mu^2
            agg_meanx2 = meanx2_all_tensor.sum(dim=0) / count
            sigma2 = agg_meanx2 - torch.pow(mu, 2)
            # 钳制最终聚合的 sigma2
            sigma2 = torch.clamp(sigma2, min=self.eps) # 确保方差非负且不小于eps
            # 更新历史缓冲区
            # 确保 .detach() 以防止历史梯度累积
            self.pre_mu = [cur_mu.detach()] + self.pre_mu[:(self.buffer_num - 1)]
            self.pre_meanx2 = [cur_meanx2.detach()] + self.pre_meanx2[:(self.buffer_num - 1)]
            self.pre_dmudw = [dmudw.detach()] + self.pre_dmudw[:(self.buffer_num - 1)]
            self.pre_dmeanx2dw = [dmeanx2dw.detach()] + self.pre_dmeanx2dw[:(self.buffer_num - 1)]

            tmp_weight = torch.zeros_like(self.weight.data)
            tmp_weight.copy_(self.weight.data)
            self.pre_weight = [tmp_weight.detach()] + self.pre_weight[:(self.buffer_num - 1)]
        else: # buffer_num == 0 或者不在训练模式，或者不需要梯度（冻结层）
            # 退化为标准 BN 或使用 running stats
            x = y # y 已经是 (C, N*H*W)
            cur_mu = x.mean(dim=1)
            cur_meanx2 = torch.pow(x, 2).mean(dim=1) # 计算当前批次的 meanx2
            cur_sigma2 = cur_meanx2 - torch.pow(cur_mu, 2)
            cur_sigma2 = torch.clamp(cur_sigma2, min=0.0) # Clamp current batch variance
            if not self.training or self.FROZEN:
                # 推理模式或冻结模式，使用 running_mean/var
                mu = self.running_mean
                sigma2 = self.running_var
                sigma2 = torch.clamp(sigma2, min=self.eps) # 确保运行方差非负且不小于eps
            else:
                # 训练模式但 buffer_num=0 (burnin 阶段)
                mu = cur_mu
                sigma2 = cur_sigma2 # 已经 clamp 过
                # 更新 running stats
                if self.track_running_stats:
                    with torch.no_grad():
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma2
        # 归一化步骤和最终仿射变换 
        # 确保方差为正且不为零，防止除以零
        std = torch.sqrt(sigma2 + self.eps)
        # 再次确保 std 不为零
        std = torch.where(std == 0, torch.full_like(std, self.eps), std)
        # 执行规范化
        normalized_y = (y - mu.view(-1, 1)) / std.view(-1, 1)
        # 执行缩放和平移
        if self.affine:
            output = self.weight.view(-1, 1) * normalized_y + self.bias.view(-1, 1)
        else:
            output = normalized_y
        # 恢复原始形状并转置回来
        # input.shape 是 (N, C, H, W)
        # output 现在是 (C, N*H*W)
        # 需要将其恢复为 (N, C, H, W)
        # original_shape = input.shape # 应该在函数开始时保存原始形状
        output = output.view(input.size(1), input.size(0), *input.shape[2:]) # C, N, H, W
        output = output.transpose(0, 1) # N, C, H, W
        # 确保输出形状与输入形状一致
        return output
    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'buffer={max_buffer_num}, burnin={burnin}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

#3.数据加载器与模型定义
def get_voc_dataloader(batch_size, voc_root='/root/autodl-tmp/VOC2007', distributed=True):# 数据加载器
    # 路径检查
    if not os.path.exists(os.path.join(voc_root, 'VOCdevkit/VOC2007')):
        raise FileNotFoundError(f"VOC2007数据集目录结构不正确，请确保{voc_root}/VOCdevkit/VOC2007存在")
    # 数据集实例化
    train_dataset = VOCDetection(
        root=voc_root,
        year='2007',
        image_set='trainval',
        download=True,#当前目录不包括VOC2007数据集，会自动下载
        transforms=VOC_Transforms(train=True) # 只传入 transforms 参数
    )
    val_dataset = VOCDetection(
        root=voc_root,
        year='2007',
        image_set='test',
        download=True,
        transforms=VOC_Transforms(train=False) # 只传入 transforms 参数
    )

    # 获取分布式环境参数
    # 如果处于分布式模式，则获取 rank 和 world_size
    # 否则，默认为单进程模式，rank=0, world_size=1
    current_rank = 0
    world_size = 1
    if distributed and dist.is_initialized(): # 只有当 distributed=True 且分布式环境已初始化时才获取
        current_rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    # 采样器选择
    if distributed:
        # shuffle 默认为 True for train_sampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=current_rank, shuffle=True)
        # 验证集通常不 shuffle
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=current_rank, shuffle=False)
    else:
        # 单机模式下使用常规采样器
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    # DataLoader 实例化
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler, # 使用 sampler
        num_workers=16,#CPU核心数，根据自己的CPU调整
        collate_fn=collate_fn,
        pin_memory=True # 加速数据传输到 GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True #
    )

    # 只在主进程打印整个训练集样本数
    if current_rank == 0:
        print(f"训练集样本数: {len(train_dataset)}")
        print("检查数据集中的样本格式 (使用 transforms.v2 后):")
    
    # 为每个进程打印它将处理的局部样本信息
    # 确保每个进程都打印自己的前几个样本，并且是它将实际读取的样本
    if current_rank == 0: 
        for i in range(min(5, len(train_dataset))):#取前5个样本
            sample_img_v2, sample_target_v2 = train_dataset[i] # 仍取原始数据集的索引
            print(f"样本{i} - 图像shape: {sample_img_v2.shape}, dtype: {sample_img_v2.dtype}")
            print(f"样本{i}的目标类型: {type(sample_target_v2)}")
            if isinstance(sample_target_v2, dict):# 检查是否是字典
                print(f"样本{i}包含的键: {list(sample_target_v2.keys())}")
                if 'boxes' in sample_target_v2 and 'labels' in sample_target_v2:# 检查是否包含 'boxes' 和 'labels'
                    print(f"目标框数量: {len(sample_target_v2['boxes'])}, 标签数量: {len(sample_target_v2['labels'])}")
                    if len(sample_target_v2['boxes']) > 0:# 确保有有效的目标框
                        print(f"第一个目标框: {sample_target_v2['boxes'][0].tolist()}")
                        print(f"第一个标签: {sample_target_v2['labels'][0].item()}")
                    else:
                        print("此样本在经过 transforms.v2 处理后没有有效目标框。")
                else:
                    print("警告: 目标字典缺少 'boxes' 或 'labels' 键。")
            else:
                print(f"警告: 不支持的目标类型: {type(sample_target_v2)}")
    return train_loader, val_loader

def get_retinanet_model(norm_type='SyncBN', num_classes=21):
    # 加载COCO预训练的RetinaNet模型
    # 注意：这将加载一个针对91个类别（COCO）的模型
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)

    # 获取FPN的输出通道数，通常为256
    in_channels = 256
    #in_channels = model.backbone.fpn.out_channels
    # 获取每个位置的anchor数量，RetinaNet通常为9
    num_anchors = 9
    #num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    # **替换分类头以匹配VOC2007的21个类别**
    # 原始的分类头中可能也有BatchNorm层，这里替换时也要考虑
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes, # 你传入的21
        norm_layer=nn.BatchNorm2d # 默认使用BatchNorm2d，后面再统一替换
    )
    # 回归头通常不需要修改类别数，但如果模型结构变化，可能也需要重新初始化以确保权重匹配
    # 这里保持默认的回归头结构，如果其内部BN层需要替换，则靠后续的遍历来处理
    # model.head.regression_head = RetinaNetRegressionHead(...)

    # **替换整个模型中的BatchNorm层**
    # 这是一个辅助函数，用于遍历所有子模块并替换BatchNorm
    def replace_all_bn(module, target_norm_type_str):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                new_norm_layer = None
                if target_norm_type_str == 'CBN':
                    new_norm_layer = CBNModule(child.num_features)
                elif target_norm_type_str == 'SyncBN':
                    new_norm_layer = nn.SyncBatchNorm(child.num_features)
                elif target_norm_type_str == 'None':
                    new_norm_layer = nn.Identity()
                
                if new_norm_layer is not None:
                    # 尝试复制预训练的BN权重
                    if hasattr(new_norm_layer, 'weight') and child.weight is not None:
                        new_norm_layer.weight.data.copy_(child.weight.data)
                    if hasattr(new_norm_layer, 'bias') and child.bias is not None:
                        new_norm_layer.bias.data.copy_(child.bias.data)
                    setattr(module, name, new_norm_layer)
                else: # 如果不支持的类型，则保持原样并递归
                    replace_all_bn(child, target_norm_type_str)
            else:
                replace_all_bn(child, target_norm_type_str)

    if norm_type == 'CBN' or norm_type == 'SyncBN' or norm_type == 'None':
        print(f"正在替换RetinaNet模型中的所有BatchNorm层为{norm_type}...")
        replace_all_bn(model, norm_type) # 对整个模型进行替换

    # 设置所有参数都可训练
    for p in model.parameters():
        p.requires_grad = True

    
    # 确认FPN参数可训练
    print("正在确保FPN参数可训练...")
    for name, p in model.backbone.fpn.named_parameters():
        if p.requires_grad is False:
            p.requires_grad = True
            print(f"已将FPN参数 {name} 的 requires_grad 设为 True")
            
    return model

#4.检查模型结构完整性与梯度传播
def check_model_structure(model):
    print("\n--- 模型结构完整性检查 ---")
    # 处理DDP包装的模型
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):# 检查是否为分布式训练
        real_model = model.module# 解包模型
        print("检测到分布式训练模式(DDP)，已解包模型")
    else:
        real_model = model# 非分布式训练则直接使用模型
    
    if hasattr(real_model, 'backbone'):# 检查模型是否包含backbone属性
        backbone = real_model.backbone
    else:
        print("!!! 警告: 模型没有backbone属性")
        return

    print("FPN输入输出通道检查:")# 验证FPN输入输出通道
    if hasattr(backbone, 'fpn'):# 检查backbone是否包含fpn属性
        for name, layer in backbone.fpn.named_children():
            if isinstance(layer, nn.Conv2d):# 检查是否为卷积层
                print(f"{name}: in_channels={layer.in_channels}, out_channels={layer.out_channels}")# 打印输入输出通道数
    else:
        print("!!! 警告: backbone没有fpn属性")

    print("\n分类头检查:")# 检查分类头和回归头
    if hasattr(real_model, 'head') and hasattr(real_model.head, 'classification_head'):
        for name, param in real_model.head.classification_head.named_parameters():
            print(f"{name}: shape={param.shape}")
    else:
        print("!!! 警告: 模型没有head或classification_head属性")
    
    print("\n回归头检查:")
    if hasattr(real_model, 'head') and hasattr(real_model.head, 'regression_head'):
        for name, param in real_model.head.regression_head.named_parameters():
            print(f"{name}: shape={param.shape}")
    else:
        print("!!! 警告: 模型没有head或regression_head属性")
    
    print("\n数据流验证:")# 检查数据流
    dummy_input = torch.randn(1, 3, 300, 300).to(next(model.parameters()).device)# 创建随机输入
    try:
        with torch.no_grad():
            features = backbone(dummy_input)
            print("FPN输出特征图尺寸:")
            for k, v in features.items():
                print(f"{k}: {v.shape}")
    except Exception as e:
        print(f"数据流验证失败: {str(e)}")

class GradCheckHook:# 梯度检查
    """
    用于监控模型训练过程中的梯度流动。
    该类会在每次前向传播和反向传播后打印输入和输出的数值统计，
    以及检查梯度是否存在NaN或Inf。
    Args:
        module (nn.Module): 要监控的PyTorch模块
        name (str): 模块的名称标识
        threshold (float): 梯度警告阈值，默认1e-6
    """
    def __init__(self, module, name, threshold=1e-6): #
        self.module = module
        self.name = name
        self.threshold = threshold
        self.grad_mean_abs = 0.0
        self.grad_counter = 0

        # 注册前向传播和反向传播的hook
        # forward_hook 会在每次前向传播后执行
        self.forward_handle = module.register_forward_hook(self._forward_hook_fn)
        # backward_hook 会在每次反向传播后执行
        self.backward_handle = module.register_full_backward_hook(self._backward_hook_fn)

    def _forward_hook_fn(self, module, input, output):#前向传播hook函数，记录输入输出统计信息
        # 记录输入特征图的数值统计
        if isinstance(input, tuple):
            for i, inp_tensor in enumerate(input):
                if torch.is_tensor(inp_tensor) and inp_tensor.requires_grad:
                    print(f"    Input (to {self.module.__class__.__name__}): "
                          f"shape={inp_tensor.shape}, "
                          f"mean={inp_tensor.mean().item():.3e}, "
                          f"std={inp_tensor.std().item():.3e}, "
                          f"min={inp_tensor.min().item():.3e}, "
                          f"max={inp_tensor.max().item():.3e}")
        elif torch.is_tensor(input) and input.requires_grad:
             print(f"    Input (to {self.module.__class__.__name__}): "
                          f"shape={input.shape}, "
                          f"mean={input.mean().item():.3e}, "
                          f"std={input.std().item():.3e}, "
                          f"min={input.min().item():.3e}, "
                          f"max={input.max().item():.3e}")
        # 记录输出特征图的数值统计
        if isinstance(output, tuple):
            for i, out_tensor in enumerate(output):
                if torch.is_tensor(out_tensor) and out_tensor.requires_grad:
                    print(f"    Output (from {self.module.__class__.__name__}): "
                          f"shape={out_tensor.shape}, "
                          f"mean={out_tensor.mean().item():.3e}, "
                          f"std={out_tensor.std().item():.3e}, "
                          f"min={out_tensor.min().item():.3e}, "
                          f"max={out_tensor.max().item():.3e}")
                    print(f"    Output grad_fn: {out_tensor.grad_fn}")
                    print(f"    Output requires_grad: {out_tensor.requires_grad} (OK)")
        elif torch.is_tensor(output) and output.requires_grad:  
            print(f"    Output (from {self.module.__class__.__name__}): "
                          f"shape={output.shape}, "
                          f"mean={output.mean().item():.3e}, "
                          f"std={output.std().item():.3e}, "
                          f"min={output.min().item():.3e}, "
                          f"max={output.max().item():.3e}")
            print(f"    Output grad_fn: {output.grad_fn}")
            print(f"    Output requires_grad: {output.requires_grad} (OK)")


    def _backward_hook_fn(self, module, grad_input, grad_output):
        # 反向传播hook函数，监控梯度情况，我们只关心可学习参数的梯度，特别是weight和bias
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_abs_mean = grad.abs().mean().item()
                # 只有当平均绝对梯度值低于阈值时才发出警告
                if grad_abs_mean < self.threshold:
                    warnings.warn(f"警告: {self.name}.{name} (BN/CBN) 梯度值过小 ({grad_abs_mean:.3e})")

                self.grad_mean_abs = (self.grad_mean_abs * self.grad_counter + grad_abs_mean) / (self.grad_counter + 1)
                self.grad_counter += 1
    def remove(self):
        #移除hook，防止内存泄漏
        self.forward_handle.remove()
        self.backward_handle.remove()

#5.评估指标
def bbox_iou(box1, box2):# 计算两个边界框的IoU
    """
    Args:
        box1: List[float] - 第一个边界框坐标[x1,y1,x2,y2]
        box2: List[float] - 第二个边界框坐标[x1,y1,x2,y2]
    Returns:
        float - IoU值
    """
    x1 = max(box1[0], box2[0]) 
    y1 = max(box1[1], box2[1])  
    x2 = min(box1[2], box2[2])  
    y2 = min(box1[3], box2[3])  

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1) # 交集面积

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) 
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) 

    iou = inter_area / float(box1_area + box2_area - inter_area) 
    return iou 

def voc_eval(detections, ground_truths, class_id, ovthresh=0.5, use_07_metric=False):
    """
    计算VOC评估指标, 计算精度（AP）。
    Args:
        detections: Dict[str, List[Dict]] - 检测结果字典
        ground_truths: Dict[str, List[Dict]] - 真实标注字典
        class_id: int - 类别ID
        ovthresh: float - IoU阈值
        use_07_metric: bool - 是否使用VOC2007评估方式
    Returns:
        Tuple[np.ndarray, np.ndarray, float] - (召回率, 精确率, AP)
    """
    class_dets = []  
    for img_id, dets in detections.items():  
        for det in dets:  
            if det['category_id'] == class_id:  # 只考虑指定类别
                class_dets.append((img_id, det['score'], det['bbox'])) # 
    class_dets.sort(key=lambda x: -x[1])  # 按置信度降序排序
    npos = 0  
    class_gts = defaultdict(list) # 存储每个图像的标注
    for img_id, gts in ground_truths.items():  # 统计所有标注
        for gt in gts:  
            if gt['category_id'] == class_id:  # 只考虑指定类别
                class_gts[img_id].append(gt)  
                if not gt['difficult']:  # 只统计非困难标注
                    npos += 1  
    tp = np.zeros(len(class_dets))  # 初始化真阳
    fp = np.zeros(len(class_dets))  # 初始化假阳
    for i, (img_id, _, det_bbox) in enumerate(class_dets): 
        if img_id not in class_gts:  # 跳过没有标注的图像
            fp[i] = 1 
            continue  
        max_iou = -np.inf  
        best_gt_idx = -1  
        gts = class_gts[img_id]  # 获取当前图像的标注

        for j, gt in enumerate(gts):  # 遍历所有标注
            iou = bbox_iou(det_bbox, gt['bbox'])  
            if iou > max_iou:  # 更新最大IoU
                max_iou = iou  
                best_gt_idx = j  # 记录最佳匹配
        if max_iou >= ovthresh: # 匹配成功
            if not gts[best_gt_idx]['used']:  # 避免重复匹配
                tp[i] = 1  
                gts[best_gt_idx]['used'] = True  # 标记为已使用
            else:  # 重复匹配，视为假阳性
                fp[i] = 1  
        else:  # 匹配失败，视为假阳性
            fp[i] = 1  

    fp = np.cumsum(fp)  # 累积假阳性
    tp = np.cumsum(tp)  # 累积真阳性
    rec = tp / float(npos)  # 召回率
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 精确率
    if use_07_metric: # 如果使用VOC2007评估方式，11点插值法
        ap = 0. 
        for t in np.arange(0., 1.1, 0.1):  
            if np.sum(rec >= t) == 0:  
                p = 0  
            else: # 
                p = np.max(prec[rec >= t])  
            ap += p / 11.  
    else: # 否则使用AP计算
        mrec = np.concatenate(([0.], rec, [1.]))  # 增加边界
        mpre = np.concatenate(([0.], prec, [0.]))  
        for i in range(mpre.size - 1, 0, -1):  # 逆序遍历
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # 插值
        i = np.where(mrec[1:] != mrec[:-1])[0]  
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  

    return rec, prec, ap  

#6.训练与推理评估
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs,
                lr_log_file_path, loss_log_file_path, mAP_log_file_path,
                model_save_path, rank=0, eval_interval=10): 
    """
    训练模型主循环
    Args:
        model: nn.Module - 待训练模型
        train_loader: DataLoader - 训练数据加载器
        val_loader: DataLoader - 验证数据加载器
        optimizer: optim.Optimizer - 优化器
        scheduler: optim.lr_scheduler - 学习率调度器
        device: torch.device - 训练设备
        num_epochs: int - 训练轮数
        lr_log_file_path: str - 学习率日志路径
        loss_log_file_path: str - 损失日志路径
        mAP_log_file_path: str - mAP日志路径
        model_save_path: str - 模型保存路径
        rank: int - 进程rank(分布式训练用)
        eval_interval: int - 评估间隔
    Returns:
        Tuple[List[float], List[float], List[float], List[float]] - 
        (损失列表, 学习率列表, mAP列表, 损失方差列表)
    """
    scaler = GradScaler() # 混合精度训练
    best_mAP = 0.0 # 初始化最佳mAP
    epoch_losses = [] #损失列表
    epoch_lrs = [] #学习率列表
    epoch_mAPs = [] #mAP列表
    epoch_loss_variances = [] #损失方差列表
    is_main_process = (rank == 0)# 检查是否为主进程

    for epoch in range(num_epochs):# 训练轮数
        model.train()# 设置模型为训练模式
        if hasattr(train_loader.sampler,'set_epoch'):# 设置分布式采样器的epoch
            train_loader.sampler.set_epoch(epoch)
        current_epoch_losses = []# 当前轮次损失列表
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]# 当前轮次学习率列表
        if is_main_process: # 只有主进程打印学习率和 epoch 信息
            print(f"Epoch {epoch+1}/{num_epochs}, Current LR: {current_lrs}")
            with open(lr_log_file_path, 'a') as f:# 记录学习率
                f.write(f"Epoch {epoch+1}: {current_lrs}\n")
        epoch_lrs.append(current_lrs[0])
        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process)
        for batch_idx, (images, targets) in enumerate(pbar):
            # images 和 targets 在这里仍然是 CPU 上的数据
            # --- 调试：原始 CPU 上的数据检查 (只在主进程和前5个批次进行) ---
            if is_main_process and (epoch == 0 and batch_idx < 5):
                print(f"\n--- DEBUG: Batch {batch_idx} targets (CPU before device transfer) ---")
                for i, target_item in enumerate(targets):# 遍历每个目标
                    print(f"  Image {i} in batch {batch_idx}:")# 打印图像索引
                    if 'boxes' in target_item and target_item['boxes'] is not None:# 检查是否有边界框
                        print(f"    Boxes shape: {target_item['boxes'].shape}, num_boxes: {target_item['boxes'].numel() // 4}")
                        if target_item['boxes'].numel() == 0:
                            print("    !!! WARNING: Boxes tensor is EMPTY for this image!")
                        if target_item['boxes'].numel() > 0:
                            print(f"    First box (xmin, ymin, xmax, ymax): {target_item['boxes'][0].tolist()}")
                    else:
                        print("    !!! WARNING: 'boxes' key missing or is None!")
                    if 'labels' in target_item and target_item['labels'] is not None:
                        print(f"    Labels shape: {target_item['labels'].shape}, num_labels: {target_item['labels'].numel()}")
                        if target_item['labels'].numel() == 0:
                            print("    !!! WARNING: Labels tensor is EMPTY for this image!")
                        if target_item['labels'].numel() > 0:
                            print(f"    First label: {target_item['labels'][0].item()}")
                    else:
                        print("    !!! WARNING: 'labels' key missing or is None!")
                print("---------------------------------------------------\n")
            # 将图像和目标移动到设备上
            images_on_device = [image.to(device) for image in images]
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
            images_batch_tensor = torch.stack(images_on_device, dim=0)#将列表中的图像张量堆叠成一个批处理张量
            # --- 调试：第一个 Batch 的数据和标签信息 (在设备上) ---
            # 确保只在主进程的第一个批次打印一次，且提供更详细的信息
            if batch_idx == 0 and is_main_process:
                print("\n--- DEBUG: First Batch Data and Target Info on Device ---")
                print(f"Images on device (len): {len(images_on_device)}")
                if len(images_on_device) > 0:
                    print(f"  First image shape: {images_on_device[0].shape}, dtype: {images_on_device[0].dtype}")
                    print(f"  First image min/max/mean/std: {images_on_device[0].min().item():.3f}/"
                          f"{images_on_device[0].max().item():.3f}/"
                          f"{images_on_device[0].mean().item():.3f}/"
                          f"{images_on_device[0].std().item():.3f}")
                print(f"Targets on device (len): {len(targets_on_device)}")
                for i, t in enumerate(targets_on_device):
                    print(f"  Target {i}:")
                    if 'boxes' in t and t['boxes'].numel() > 0:
                        print(f"    boxes_shape={t['boxes'].shape}, boxes_dtype={t['boxes'].dtype}")
                    else:
                        print(f"    WARNING: Target {i} has no bounding boxes (empty)! Numel: {t['boxes'].numel()}")
                    if 'labels' in t and t['labels'].numel() > 0:
                        print(f"    labels_shape={t['labels'].shape}, labels_dtype={t['labels'].dtype}")
                    else:
                        print(f"    WARNING: Target {i} has no labels (empty)! Numel: {t['labels'].numel()}")
                    if t['boxes'].numel() > 0 and (t['boxes'].isnan().any() or t['boxes'].isinf().any()):
                        print(f"    CRITICAL: Target {i} boxes contain NaN/Inf!")
                    if t['labels'].numel() > 0 and (t['labels'].isnan().any() or t['labels'].isinf().any()):
                        print(f"    CRITICAL: Target {i} labels contain NaN/Inf!")
                print("--- DEBUG: First Batch info printed. ---")
            optimizer.zero_grad()# 梯度清零
            with autocast(): # 使用混合精度训练需要。
                loss_dict = model(images_batch_tensor, targets_on_device) # **确保使用 images_batch_tensor**
                losses = sum(loss for loss in loss_dict.values())
            # 检查损失是否为 NaN 或 Inf ---
            if is_main_process and (losses.isnan().any() or losses.isinf().any()):
                print(f"!!! WARNING: NaN or Inf loss detected at batch {batch_idx}, epoch {epoch+1}.")
                for k, v in loss_dict.items():
                    print(f"    Loss component '{k}': {v.item():.4f}") # 打印各个损失分量
            scaler.scale(losses).backward() # 使用 scaler.scale 进行反向传播
            # 检查梯度是否为 NaN 或 Inf (仅在主进程和第一个批次) ---
            if is_main_process and epoch == 0 and batch_idx == 0:
                print("\n--- DEBUG: NaN/Inf Gradient Check (First Batch) ---")
                model_to_check_grad = model.module if hasattr(model, 'module') else model
                for name, param in model_to_check_grad.named_parameters():
                    if param.grad is not None:
                        if param.grad.isnan().any() or param.grad.isinf().any():# 检查梯度是否为 NaN 或 Inf
                            print(f"!!! WARNING: NaN or Inf gradient detected in {name}.")
                    elif param.requires_grad: # 如果需要梯度但 grad 为 None
                        print(f"    WARNING: Parameter '{name}' requires grad but grad is None.")
                print("-------------------------------------------\n")
            # 梯度裁剪 (确保只在主进程中打印相关信息)
            scaler.unscale_(optimizer) # 由于使用了混合精度，在裁剪前，先unscale梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # 裁剪整个模型的梯度
            scaler.step(optimizer) # 使用 scaler.step 更新参数
            scaler.update() # 更新 scaler
            # --- DEBUG: 检查模型参数是否为 NaN 或 Inf (仅在主进程和第一个批次) ---
            if is_main_process and epoch == 0 and batch_idx == 0:
                print("\n--- DEBUG: NaN/Inf Parameter Check (First Batch) ---")
                model_to_check_param = model.module if hasattr(model, 'module') else model
                for name, param in model_to_check_param.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"!!! WARNING: NaN or Inf parameter detected in {name}.")
                        # print(f"    Parameter value: {param.data}") # 详细打印
                print("-------------------------------------------\n")
            # 更新进度条损失信息 (在所有进程上调用，但只有主进程会实际显示)
            pbar.set_postfix(loss=losses.item(), refresh=True)
            current_epoch_losses.append(losses.item()) # 将损失添加到列表
        # epoch 结束后的操作 (确保只在主进程中执行)
        if is_main_process:
            avg_epoch_loss = np.mean(current_epoch_losses) if current_epoch_losses else float('nan') # 防止空列表
            epoch_losses.append(avg_epoch_loss)
            if len(current_epoch_losses) > 1:
                epoch_var_loss = np.var(current_epoch_losses)
            else:
                epoch_var_loss = 0.0 # 如果只有一个损失，方差为0
            epoch_loss_variances.append(epoch_var_loss)
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Loss Variance: {epoch_var_loss:.4e}")

            with open(loss_log_file_path, 'a') as f:
                f.write(f"Epoch {epoch+1}: Avg Loss={avg_epoch_loss:.4f}, Loss Variance={epoch_var_loss:.4e}\n")
            if scheduler is not None:# 如果使用学习率调度器
                scheduler.step()

            # 每隔 eval_interval 个 epoch 评估一次模型
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                model.eval() # 切换到评估模式
                # 评估模型时，通常不需要 GradScaler
                # evaluate_model 函数也应该接收 images_batch_tensor
                mAP, fps = evaluate_model(model, val_loader, device) # 假设 evaluate_model 接受 model

                print(f"Validation mAP: {mAP:.4f}, FPS: {fps:.2f}")

                epoch_mAPs.append(mAP)
                with open(mAP_log_file_path, 'a') as f:
                    f.write(f"Epoch {epoch+1}: mAP={mAP:.4f}, FPS={fps:.2f}\n")
                # 保存最佳模型
                if mAP > best_mAP:
                    best_mAP = mAP
                    save_model = model.module if hasattr(model, 'module') else model
                    torch.save(save_model.state_dict(), os.path.join(model_save_path, f'best_model_epoch_{epoch+1}_mAP_{mAP:.4f}.pth'))
                    print(f"Best model saved with mAP: {best_mAP:.4f}")
                model.train() # 切换回训练模式
            else:
                epoch_mAPs.append(np.nan) # 或者记录上一次的mAP，或者不记录
    return epoch_losses, epoch_lrs, epoch_mAPs, epoch_loss_variances # 返回方差

def evaluate_model(model, val_loader, device):
    """
    评估模型性能
    Args:
        model: nn.Module - 待评估模型
        val_loader: DataLoader - 验证数据加载器
        device: torch.device - 评估设备
    Returns:
        Tuple[float, float] - (mAP, FPS)
    """
    model.eval() # 
    detections = defaultdict(list) # 
    ground_truths = defaultdict(list) # 
    start_time = time.time() # 
    with torch.no_grad(): # 
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)): # 
            if len(images) == 0: # 
                continue # 
            images = list(image.to(device) for image in images) # 
            outputs = model(images) # 
            for i, t in enumerate(targets): # 
                img_id = f"{batch_idx}_{i}" # 
                boxes = t['boxes'].cpu().numpy() # 
                labels = t['labels'].cpu().numpy() # 
                for box, label in zip(boxes, labels): # 
                    if 1 <= label <= len(VOC_CLASSES): # 
                        ground_truths[img_id].append({ # 
                            'bbox': box.tolist(), # 
                            'category_id': int(label), # 
                            'difficult': False,  
                            'used': False  
                        })
            for i, output in enumerate(outputs):  
                img_id = f"{batch_idx}_{i}"  
                boxes = output['boxes'].cpu().numpy()  
                scores = output['scores'].cpu().numpy()  
                labels = output['labels'].cpu().numpy()  
                for box, score, label in zip(boxes, scores, labels):  
                    if score > 0.01: # 
                        if 1 <= label <= len(VOC_CLASSES):  
                            detections[img_id].append({  
                                'bbox': box.tolist(),  
                                'score': float(score),  
                                'category_id': int(label)  
                            })
    aps = [] # 
    for cls_id in range(1, len(VOC_CLASSES) + 1):  
        rec, prec, ap = voc_eval(detections, ground_truths, cls_id) 
        aps.append(ap)  # 计算每个类别AP
    mAP = np.mean(aps)  
    if math.isnan(mAP):
        mAP = 0.0 #mAP为NaN时，将其设置为0.0，避免后续计算错误
    end_time = time.time()  
    total_images_processed = 0  
    for images, _ in val_loader:  
        total_images_processed += len(images)  
    fps = total_images_processed / (end_time - start_time) if (end_time - start_time) > 0 else 0.0  
    return mAP, fps

#7.主函数 
def main():
    if not os.path.exists('VOC2007/VOCdevkit/VOC2007'): # 检查数据集是否存在 
        print("请先下载VOC2007数据集并解压到VOC2007目录")  
        return  

    use_distributed = 'RANK' in os.environ # 检查是否使用分布式训练 
    rank = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 选择设备 
    if use_distributed:
        torch.distributed.init_process_group(backend='nccl')
        rank = int(os.environ['LOCAL_RANK']) 
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        print(f"Running in distributed mode. Rank: {rank}, Device: {device}") # 打印当前进程的 rank
    else:
        print("不使用分布式训练")
    create_folder_structure() # 创建文件夹结构

    batch_sizes = [8] 
    num_epochs = 100 # 训练轮数 
    initial_learning_rate = 0.0001 #初始学习率

    scheduler_type = 'CosineAnnealingLR' #余弦退火，训练过程中，学习率随迭代次数按余弦函数规律衰减。
    T_max =200 #迭代次数
    eta_min = 1e-8 #学习率最小值 
    # scheduler_type = 'StepLR' #不同学习率调度器 
    # step_size = 20 
    # gamma = 0.1 
    #scheduler_type = None 

    models_info = { # 模型信息字典，包含模型函数、优化器和损失函数 
        'RetinaNet': {  
            'model_func': get_retinanet_model,  
            'optimizer': optim.Adam,  #使用Adam优化器
            'criterion': None  # 由于RetinaNet使用了自定义的损失函数，这里不指定
        },
    }
    norm_types = ['CBN','SyncBN','None'] #不同规范化类型 
    for batch_size in batch_sizes:#便于训练不同batch_size
        # 在这里获取一次 DataLoader，它会在每次 model_name 和 norm_type 循环时重用
        train_loader, val_loader = get_voc_dataloader(batch_size, distributed=use_distributed) # 获取数据加载器
        for model_name, info in models_info.items():
            for norm_type in norm_types:
                print(f"\n--- Starting training for Model: {model_name}, Norm: {norm_type}, Batch Size: {batch_size}, LR: {initial_learning_rate} ---")
                # 获取模型并移动到设备

                model = info['model_func'](norm_type).to(device)
                if use_distributed:# 如果使用分布式训练
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
                # 修改优化器设置
                params = [
                    {'params': [p for n, p in model.named_parameters() if 'backbone.fpn' not in n], 'lr': initial_learning_rate},
                    # 可选择为FPN部分设置更大的学习率
                    {'params': [p for n, p in model.named_parameters() if 'backbone.fpn' in n], 'lr': initial_learning_rate * 1}
                ]
                optimizer = info['optimizer'](params, lr=initial_learning_rate)# 使用Adam优化器
                scheduler = None # 初始化调度器为 None
                if scheduler_type == 'StepLR': # 如果使用 StepLR 调度器
                    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                elif scheduler_type == 'CosineAnnealingLR':# 如果使用余弦退火调度器
                    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
                # 构建学习率、损失、mAP日志文件和模型保存的路径
                current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                # 确保路径一致性
                log_base_dir = os.path.join('CBN_COCO', model_name, norm_type, f'bs_{batch_size}_lr_{initial_learning_rate}_{current_time_str}')
                os.makedirs(log_base_dir, exist_ok=True) # 确保目录存在
                lr_log_file_path = os.path.join(log_base_dir, f'lr_log.txt')
                loss_log_file_path = os.path.join(log_base_dir, f'loss_log.txt')
                mAP_log_file_path = os.path.join(log_base_dir, f'mAP_log.txt')
                model_save_path = os.path.join(log_base_dir, f'saved_models')
                os.makedirs(model_save_path, exist_ok=True) # 创建模型保存目录
                epoch_losses, epoch_lrs, epoch_mAPs, epoch_loss_variances = train_model(
                    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs,
                    lr_log_file_path, loss_log_file_path, mAP_log_file_path,
                    model_save_path, rank=0, eval_interval=5 # 每5个epoch评估一次
                )
                # 保存训练历史数据（Loss, LR, mAP, Loss Variance）为 JSON
                history_data = {
                    'epoch_losses': epoch_losses,
                    'epoch_lrs': epoch_lrs,
                    'epoch_mAPs': epoch_mAPs,
                    'epoch_loss_variances': epoch_loss_variances # 添加方差
                }
                def clean_nan_for_json(obj):
                    if isinstance(obj, list):
                        return [clean_nan_for_json(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: clean_nan_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, float) and math.isnan(obj):
                        return None # 将 NaN 替换为 None，JSON会将其序列化为 null
                    else:
                        return obj
                    # 在写入 JSON 之前调用清理函数
                cleaned_history_data = clean_nan_for_json(history_data)
                history_file_path = os.path.join(log_base_dir, f'training_history.json')
                with open(history_file_path, 'w') as f:
                    json.dump(cleaned_history_data , f, indent=4)
                print(f"训练历史数据已保存到 {history_file_path}")
                # 最终结果文件的生成
                final_mAP = epoch_mAPs[-1] if epoch_mAPs and not np.isnan(epoch_mAPs[-1]) else 0.0 # 取最后一个有效mAP
                final_loss = epoch_losses[-1] if epoch_losses else 0.0 # 取最后一个epoch的平均loss
                final_loss_variance = epoch_loss_variances[-1] if epoch_loss_variances else 0.0 # 取最后一个epoch的损失方差
                # 确保 eval_model 可以被访问，并获取 FPS (如果不在DDP主进程，则可能不需要重新评估)
                if rank == 0: # 只有主进程进行最终评估和结果文件写入
                    mAP_final_eval, fps = evaluate_model(model, val_loader, device) # 重新评估一次获取最终FPS和mAP
                else:
                    mAP_final_eval, fps = 0.0, 0.0 # 非主进程不评估
                scheduler_info_str = "" #初始化调度器信息字符串
                if scheduler_type == 'StepLR':
                    scheduler_info_str = f"_sch-step_ss-{step_size}_g-{gamma}"
                elif scheduler_type == 'CosineAnnealingLR':
                    scheduler_info_str = f"_sch-cosine_tm-{T_max}_emin-{eta_min}"
                # 最终结果文件应该在 log_base_dir 下
                result_file = os.path.join(log_base_dir,
                                            f'results_lr_{initial_learning_rate}_batch_{batch_size}_epoch_{num_epochs}{scheduler_info_str}.txt')
                with open(result_file, 'w') as f:# 写入最终结果到文件
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
                    f.write(f'Final mAP (from final eval): {mAP_final_eval:.4f}\n') # 修改为最终mAP
                    f.write(f'FPS: {fps:.2f}\n')
                    f.write(f'Final Epoch Average Loss: {final_loss:.4f}\n') # 添加最终平均损失
                    f.write(f'Final Epoch Loss Variance: {final_loss_variance:.4e}\n') # 添加最终损失方差
                print(f"Results saved to {result_file}")
    if use_distributed and rank == 0:
        dist.destroy_process_group() # 使用分布式训练需要销毁进程组，确保资源释放
        print("DDP process group destroyed.")

if __name__ == "__main__":
    main()