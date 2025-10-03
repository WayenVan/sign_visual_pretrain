import hydra

from omegaconf import DictConfig, OmegaConf
import os
from accelerate import Accelerator
from transformers import set_seed

from ..modeling_sign_visual.sign_pt_model import (
    SignVisualModelForPretrain,
)
from ..configuration_sign_visual.configuration import SignPretrainConfig
from ..engine.training_args import SignPretrainTrainingArguments
from ..engine.trainer import SignPTTrainer
from ..data.datamodule import DataModule


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(
    version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="base_train"
)
def main(cfg: DictConfig):
    # accelerate initialize
    acc = Accelerator()

    # create model
    config = SignPretrainConfig(
        **OmegaConf.to_container(cfg.model.config, resolve=True)
    )
    model = SignVisualModelForPretrain(config)

    datamodule = DataModule(cfg.data)
    datamodule.setup("train")

    # create trainer
    training_args = SignPretrainTrainingArguments(
        **cfg.engine.training_args,
    )
    trainer = SignPTTrainer(
        model=model,
        args=training_args,
        hydra_config=cfg,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.val_dataset,
        data_collator=datamodule.collator,
    )

    # trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
