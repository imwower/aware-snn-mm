#!/usr/bin/env bash
set -e

# Run vision unsupervised STDP
python -m snnmm.training.train_unsup_vision --data-root data/cifar-100-python --epochs 1 --batch-size 64 --num-workers 4 --timesteps 20 --stage-depth 3 --stage1-size 256 --stage2-size 256 --stage3-size 128

# Run text unsupervised STDP
python -m snnmm.training.train_unsup_text --data-root data/cifar-100-python --epochs 1 --batch-size 128 --num-workers 2 --timesteps 20 --high-rate 0.9 --low-rate 0.01

# Align core (awareness core + gate adjustment)
python -m snnmm.training.train_align_core --config configs/cifar100_align_core.yaml

# Run R-STDP classifier
python -m snnmm.training.train_rstdp_classifier --data-root data/cifar-100-python --epochs 1 --batch-size 32 --num-workers 2 --timesteps 20 --cycle-length 10
