export WANDB_PROJECT=sign_language_visual_pretrain

export PYTHONPATH=./src:$PYTHONPATH

accelerate launch --num_processes=2 --mixed_precision=fp16 \
	-m csi_sign_pretrain.commands.train \
	model=convnext \
	engine.training_args.auto_output_root=./outputs/info_nce_pretrain \
	engine.training_args.num_train_epochs=1000 \
	engine.training_args.per_device_train_batch_size=256
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# 	engine.training_args.auto_output_root=./outputs/first_demo_ft
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# 	-m csi_slt.commands.train_ft \
# 	engine.training_args.auto_output_root=./outputs/first_demo_ft
