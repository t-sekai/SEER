# Stored Embeddings for Efficient Reinforcement Learning (SEER)

Official codebase for [Improving Computational Efficiency in Visual Reinforcement Learning via Stored Embeddings](https://arxiv.org/abs/2103.02886). The Rainbow codebase was originally forked from Kaixhin's [Rainbow](https://github.com/Kaixhin/Rainbow) and the CURL codebase was originally forked from [CURL](https://github.com/MishaLaskin/curl).

## BibTex

```
```

# Rainbow + SEER

## Instructions
See instructions in Kaixhin's [Rainbow](https://github.com/Kaixhin/Rainbow). Additional hyperparameters are steps_until_freeze. Example scripts for using large replay buffers (run_lb_alien.sh) and small replay buffers (run_sb_alien.sh) can be found in the scripts folder.

# CURL + SEER

## Instructions
See instructions in [CURL](https://github.com/MishaLaskin/curl). Additional hyperparameters are steps_until_freeze and num_copies. Example scripts for using large replay buffers (run_lb_cartpole.sh) and small replay buffers (run_sb_cartpole.sh) can be found in the scripts folder.

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --action_repeat 8 \
    --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp \
    --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 
```
