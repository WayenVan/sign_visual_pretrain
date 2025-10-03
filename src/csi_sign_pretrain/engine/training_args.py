import os
from dataclasses import dataclass, field
from datetime import datetime

from transformers.training_args import TrainingArguments
from transformers.utils import logging

from accelerate import Accelerator
import torch


logger = logging.get_logger(__name__)


@dataclass
class SignPretrainTrainingArguments(TrainingArguments):
    auto_output_dir: bool = field(
        default=True,
        metadata={
            "help": "Whether to automatically set the output directory based on the model name."
            " If set to False, the output directory will be set to the value of `output_dir`."
        },
    )
    auto_output_root: str | None = field(
        default=None,
        metadata={
            "help": "The root directory to save the training results to when `auto_output_dir` is True."
            " The final output directory will be `<auto_output_root>/<model_name>`."
        },
    )

    alpha_teacher: float = field(
        default=0.99,
        metadata={
            "help": "The momentum coefficient for updating the teacher model and the feature center."
            " Typical values are in the range [0.9, 0.999]."
        },
    )

    alpha_center: float = field(
        default=0.9,
        metadata={
            "help": "The momentum coefficient for updating the feature center."
            " Typical values are in the range [0.9, 0.99]."
        },
    )

    # temperature_student: float = field(
    #     default=0.1,
    #     metadata={
    #         "help": "The temperature parameter for sharpening the student model's output distribution."
    #         " Typical values are in the range [0.04, 0.2]."
    #     },
    # )
    temperature_teacher: float = field(
        default=0.04,
        metadata={
            "help": "The temperature parameter for sharpening the teacher model's output distribution."
            " Typical values are in the range [0.04, 0.2]."
        },
    )

    @staticmethod
    def __init_output_base_name():
        now = datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def __snyc_output_base_name(acc: Accelerator, base_name: str):
        """
        synchronize the base name of the output directory across all processes.
        through the bytes communication of torch tensors.
        """
        with torch.no_grad():
            if acc.is_main_process:
                bytes_base_name = base_name.encode("utf-8")
                length = torch.tensor(
                    len(bytes_base_name), dtype=torch.long, device=acc.device
                )
            else:
                length = torch.tensor(0, dtype=torch.long, device=acc.device)

            length = acc.gather(length)
            if length.dim() > 0:
                length = length.max().cpu().item()
            else:
                length = length.cpu().item()

            content = torch.zeros(length, dtype=torch.uint8, device=acc.device)
            if acc.is_main_process:
                content = torch.tensor(
                    list(bytes_base_name), dtype=torch.uint8, device=acc.device
                )
            content = acc.gather(content)[:length].cpu().numpy().tobytes()
        return content.decode("utf-8")

    def __post_init__(self):
        super().__post_init__()

        acc = Accelerator()

        if self.auto_output_dir:
            if not self.auto_output_root:
                raise ValueError(
                    "auto_output_root must be specified when auto_output_dir is True."
                )

            base_name = self.__init_output_base_name()
            base_name = self.__snyc_output_base_name(acc, base_name)

            output_dir = os.path.join(self.auto_output_root, base_name)

            if self.output_dir:
                logger.warning(
                    "The `output_dir` argument is set, but `auto_output_dir` is True. "
                    f"The `output_dir` will be overridden from {self.output_dir} to {output_dir}."
                )
            self.output_dir = output_dir

            if self.run_name:
                logger.warning(
                    "The `run_name` argument is set, but `auto_output_dir` is True. "
                    f"The `run_name` will be overridden to the new output directory name: {output_dir}."
                )
            self.run_name = output_dir
        else:
            if self.auto_output_root:
                raise ValueError(
                    "auto_output_root must not be specified when auto_output_dir is False. "
                    "The `output_dir` will be set to the value of `output_dir`."
                )
