# Getting Started

This page provides basic tutorials about the usage of MMSceneGraph.
For installation instructions, please see [INSTALL.md](INSTALL.md).

## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset, e.g.,  coco (for detection), caption\_coco (for captioning),
visualgenome (for scene graph generation), etc.,
and also some high-level apis for easier integration to other projects.

### Test a dataset

- [x] single GPU testing
- [x] multiple GPU testing
- [x] visualize the scene graph

You can use the following commands to test a dataset.

```shell
# single-gpu testing, add the --relation_mode to visualize the scene graph
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--relation_mode]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--relation_mode]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file. It will automatically 
be saved under the working directory set in the `CONFIG_FILE`.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `predcls`, `sgcls`, `sgdet` for `visualgenome`.
- `--show`: If specified, results will be plotted on the images and saved under the `visualization` folder under the working directory.
If you would like to evaluate the dataset, do not specify `--show` at the same time.

Examples:

Assume that you have already have a checkpoint.

1. Test topic scene graph generation.

```shell
python tools/test.py configs/relcaption/VGGN_updowncrossattn_predcls_max_sous_xmask_soft_faster_RCNN_x101_64x4d.py \
    [path_to_your_ckpt].pth \
    --relcaption_mode \
    --eval predcls \
    --out [RESULT_FILE]
```

2. Test scene graph generation on visual genome (use --show to save the visualized results).
```shell
python tools/test.py configs/scene_graph/VG_PredCls_heth_area_mask_X_rcnn_x101_64x4d_fpn_1x.py \
    checkpoints/SOME_CHECKPOINT.pth \
    --eval predcls \
    --relation_mode \
    --out [RESULT_FILE] \
    --show
```

3. Test scene graph generation with 4 GPUs.

```shell
./tools/dist_test.sh configs/scene_graph/VG_PredCls_heth_area_mask_X_rcnn_x101_64x4d_fpn_1x.py \
    checkpoints/SOME_CHECKPOINT.pth \
    4 --relation_mode --out results.pkl --eval predcls
```


## Train a model

MMSceneGraph implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.
```python
evaluation = dict(interval=12)  # This evaluate the model per 12 epoch.
```

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 1, which can be modified like [this](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn_r50_fpn_1x.py#L174)) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

Difference between `resume_from` and `load_from`:
`resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load_from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

If you use launch training jobs with slurm, you need to modify the config files (usually the 6th line from the bottom in config files) to set different communication ports.

In `config1.py`,
```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`,
```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with `config1.py` ang `config2.py`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} 4
```


For more information on how it works, you can refer to [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md).
