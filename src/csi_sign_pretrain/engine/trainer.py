import torch
from transformers.trainer import Trainer
from copy import deepcopy
import torch.nn.functional as F
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from .callbacks import ModelInfoCallback, _TrainerCallbackHandler

from .callbacks import (
    SaveBestMetricCallback,
    LogHydraConfigCallback,
    SaveGitInfoCallback,
    SaveBaseModelInPEFT,
    SaveHydraConfigCallback,
)


class SignPTTrainer(Trainer):
    def __init__(self, hydra_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_teacher()
        self.callback_handler = _TrainerCallbackHandler(
            self,
            self.callback_handler.callbacks,  # WARN: replaceing the original callback handler
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )
        self.add_callback(ModelInfoCallback())
        self.add_callback(SaveGitInfoCallback())
        self.add_callback(SaveHydraConfigCallback(hydra_config))
        self.add_callback(LogHydraConfigCallback(hydra_config))

        if self.args.remove_unused_columns:
            raise ValueError(
                "remove_unused_columns should be set to False for SignPTTrainer."
            )

    def _initialize_teacher(self):
        self.teacher_model = deepcopy(self.accelerator.unwrap_model(self.model)).to(
            self.accelerator.device
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_teacher(self):
        student_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        teacher_state_dict = self.teacher_model.state_dict()
        for key in student_state_dict.keys():
            teacher_state_dict[key] = (
                self.args.alpha_teacher * teacher_state_dict[key]
                + (1 - self.args.alpha_teacher) * student_state_dict[key]
            )
        self.teacher_model.load_state_dict(teacher_state_dict)

    @torch.no_grad()
    def _update_center(self, new_center):
        unwraped_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(unwraped_model, "center"):
            raise ValueError("Model does not have 'center' attribute.")
        unwraped_model.register_buffer("center", new_center)

    def dino_loss(
        self,
        student_feats,
        teacher_feats,
        temperature_student=0.1,
        temperature_teacher=0.04,
        center_momentum=0.9,
        center=None,
    ):
        """
        计算 DINO 损失函数。

        参数:
            student_feats (torch.Tensor): 学生网络的输出特征，形状为 (batch_size, feature_dim)
            teacher_feats (torch.Tensor): 教师网络的输出特征，形状为 (batch_size, feature_dim)
            temperature (float): 锐化温度参数，默认值为 0.1
            center_momentum (float): 中心化动量，默认值为 0.9
            center (torch.Tensor, optional): 先前的教师中心，形状为 (feature_dim,)，默认为 None

        返回:
            torch.Tensor: 计算得到的损失值
            torch.Tensor: 更新后的教师中心
        """
        # L2 归一化特征
        student_feats = F.normalize(student_feats, dim=-1, p=2)
        teacher_feats = F.normalize(teacher_feats.detach(), dim=-1, p=2)

        # 中心化教师特征
        if center is None:
            center = torch.zeros_like(teacher_feats.mean(dim=0))
        center = center * center_momentum + self.accelerator.gather(teacher_feats).mean(
            dim=0
        ) * (1 - center_momentum)
        teacher_feats = teacher_feats - center

        # 锐化教师输出
        teacher_logits = teacher_feats / temperature_teacher
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # 学生输出
        student_logits = student_feats / temperature_student
        student_probs = F.softmax(student_logits, dim=-1)

        # 计算交叉熵损失
        loss = -(teacher_probs * student_probs.log()).sum(dim=-1).mean()

        return loss, center

    def distributed_info_nce_loss(self, q, k, temperature=0.07):
        """
        分布式 InfoNCE 损失
        q: query 特征, shape [B, C]
        k: key 特征, shape [B, C]
        temperature: 缩放系数
        """
        q = F.normalize(q, dim=1)
        k = F.normalize(k.detach(), dim=1)

        k_all = self.accelerator.gather(k)  # [B_total, C]

        # 2. 正样本相似度 (query 与对应 key)
        pos_logits = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # [B,1]

        # 3. 负样本相似度 (query 与所有进程的 key)
        neg_logits = torch.einsum("nc,kc->nk", [q, k_all])  # [B, B_total]

        # 4. 排除自己的正样本在负样本中
        rank = self.accelerator.process_index
        B = q.shape[0]
        idx = rank * B + torch.arange(B, device=q.device)
        neg_logits[torch.arange(B), idx] = float("-inf")  # 避免重复正样本

        # 5. 合并正负样本
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1+B_total]
        logits /= self.args.temperature_teacher

        # 6. 标签
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        # 7. 计算交叉熵
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # unwraped_model = self.accelerator.unwrap_model(model)
        anchors = inputs.pop("anchors")
        positives = inputs.pop("positives")

        # teacher forward
        with torch.no_grad():
            teacher_feats = self.teacher_model(positives, **inputs).feats

        # student forward
        student_feats = model(anchors, **inputs).feats

        # loss, new_center = self.dino_loss(
        #     student_feats,
        #     teacher_feats,
        #     center=unwraped_model.get_buffer("center"),
        #     temperature_student=self.args.temperature_student,
        #     temperature_teacher=self.args.temperature_teacher,
        #     center_momentum=self.args.alpha_center,
        # )
        # self._update_center(new_center)
        loss = self.distributed_info_nce_loss(
            student_feats, teacher_feats, temperature=0.07
        )

        if return_outputs:
            return loss, student_feats, teacher_feats

        return loss

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        self._update_teacher()
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[tuple[torch.Tensor, torch.Tensor]],
        Optional[tuple[torch.Tensor, torch.Tensor]],
    ]:
        anchors = inputs.pop("anchors")
        positives = inputs.pop("positives")

        # teacher forward
        with torch.no_grad():
            teacher_feats = self.teacher_model(positives, **inputs)

        # student forward
        student_feats = model(anchors, **inputs)

        loss = self._loss_fn(teacher_feats, student_feats)

        if prediction_loss_only:
            return loss, None, None

        return loss, (student_feats, teacher_feats), (anchors, positives)
