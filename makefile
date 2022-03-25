DATASET=
GPUS=
train:
	python -m torch.distributed.launch --nproc_per_node=$(GPUS) --use_env train.py -- --dataset=$(DATASET)
test:
	python Yelp.py --distributed=False --evaluate --checkpoint='output/Yelp/checkpoint_best.pth'
