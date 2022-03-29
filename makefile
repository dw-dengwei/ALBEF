DATASET=
GPUS=
train:
	python -m torch.distributed.launch --nproc_per_node=$(GPUS) --use_env train.py -- --dataset=$(DATASET)
test:
	python -m torch.distributed.launch --nproc_per_node=$(GPUS) --use_env train.py -- --dataset=$(DATASET) \
	--checkpoint='output/$(DATASET)/checkpoint_best.pth' --dataset=$(DATASET) --evaluate
