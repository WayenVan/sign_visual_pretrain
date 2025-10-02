import torch
from transformers.trainer import Trainer
from copy import deepcopy
import torch.nn.functional as F


class SignPTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_teacher()

    def _initialize_teacher(self):
        self.teacher_model = deepcopy(self.accelerator.unwrap_model(self.model)).to(
            self.accelerator.device
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _update_teacher(self):
        with torch.no_grad():
            student_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            teacher_state_dict = self.teacher_model.state_dict()
            for key in student_state_dict.keys():
                teacher_state_dict[key] = (
                    self.args.alpha_teacher * teacher_state_dict[key]
                    + (1 - self.args.alpha_teacher) * student_state_dict[key]
                )
            self.teacher_model.load_state_dict(teacher_state_dict)

    @torch.no_grad()
    def _update_center(self, teacher_feats):
        unwraped_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(unwraped_model, "center"):
            raise ValueError("Model does not have 'center' attribute.")

        unwraped_model.center = self.args.alpha_teacher * unwraped_model.center + (
            1 - self.args.alpha_teacher
        ) * self.accelerator.gather(teacher_feats).mean(dim=0, keepdim=True)
        return unwraped_model.center.detach().clone()

    def _loss_fn(self, teacher_feats, student_feats):
        z_t = teacher_feats  # backbone + projection head
        z_s = student_feats  # backbone + projection head

        # update center
        center = self._update_center(z_t)

        with torch.no_grad():
            z_t_centered = z_t - center
            z_t_norm = F.normalize(z_t_centered, dim=-1).detach()

        # Student forward
        z_s_norm = F.normalize(z_s, dim=-1)

        # 对齐
        return ((z_s_norm - z_t_norm) ** 2).mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        anchors = inputs.pop("anchors")
        positives = inputs.pop("positives")

        # teacher forward
        with torch.no_grad():
            teacher_feats = self.teacher_model(positives, **inputs)

        # student forward
        student_feats = model(anchors, **inputs)

        loss = self._loss_fn(teacher_feats, student_feats)

        if return_outputs:
            return loss, student_feats

        return loss
