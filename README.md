# Physics-based Indirect Illumination for Inverse Rendering



## Preparation

- Set up the python environment

```sh
conda create -n dip python=3.9
conda activate dip

pip install -r requirement.txt
```

- Download [MII](https://github.com/zju3dv/invrender) synthetic dataset from [Google Drive](https://drive.google.com/file/d/1wWWu7EaOxtVq8QNalgs6kDqsiAm7xsRh/view?usp=sharing)

## Run the code

<!--I am still cleaning my code from [![State-of-the-art Shitcode](https://img.shields.io/static/v1?label=State-of-the-art&message=Shitcode&color=7B5804)](https://github.com/trekhleb/state-of-the-art-shitcode), but you can just run the code using the following command. I changed some variables, which may lead to some bugs and can be fixed with several changes to the variables' names.-->

#### Training

Taking the scene `hotdog` as an example, the training process is as follows.

1. Pre-train the geometry.

   ```sh
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10000 --nproc_per_node=1 training/exp_runner.py --conf confs_sg/default.conf --data_split_dir [dataset_dir/hotdog] --expname hotdog --trainstage geometry --exp_dir [exp_dir]
   ```

2. Jointly optimize geometry, material, and illumination.

   ```sh
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10000 --nproc_per_node=1 training/exp_runner.py --conf confs_sg/default.conf --data_split_dir [dataset_dir/hotdog] --expname hotdog --trainstage DIP --exp_dir [exp_dir] --if_indirect --if_silhouette --unet
   ```

#### Testing

```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10000 --nproc_per_node=1 scripts/relight.py --conf confs_sg/default.conf --data_split_dir [dataset_dir/hotdog] --expname hotdog --timestamp latest --exp_dir [exp_dir] --trainstage DIP --if_indirect --unet
```

## Checkpoints

 Just in case I accidentally delete everything with rm...

* [geometry initialization](https://github.com/denghilbert/DIP/tree/main/checkpoints/geometry_initialization)
* [joint training of inverse rendering](https://github.com/denghilbert/DIP/tree/main/checkpoints/joint_train)



Acknowledgements: part of our code is inherited from  [IDR](https://github.com/lioryariv/idr), [PhySG](https://github.com/Kai-46/PhySG), and [MII](https://github.com/zju3dv/invrender). We are grateful to the authors for releasing their code.

