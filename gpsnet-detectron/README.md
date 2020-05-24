# GPS-Net: Graph Property Sensing Network for Scene Graph Generation
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/taksau/GPS-Net/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-0.4.1-%237732a8) 

This is a PyTorch implementation for [GPS-Net: Graph Property Sensing Network for Scene Graph Generation (CVPR 2020)](https://arxiv.org/pdf/2003.12962) with the backbone of [RelDN](https://github.com/NVIDIA/ContrastiveLosses4VRD).


## Requirements
* Python 3
* Python packages
  * pytorch 0.4.0 or 0.4.1.post2 (not guaranteed to work on newer versions)
  * cython
  * matplotlib
  * numpy
  * scipy
  * opencv
  * pyyaml
  * packaging
  * [pycocotools](https://github.com/cocodataset/cocoapi)
  * tensorboardX
  * tqdm
  * pillow
  * scikit-image
* An NVIDIA GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.

An easy installation if you already have Anaconda Python 3 and CUDA 9.0:
```
conda install pytorch=0.4.1
pip install cython
pip install matplotlib numpy scipy pyyaml packaging pycocotools tensorboardX tqdm pillow scikit-image
conda install opencv
```

## Compilation
Compile the CUDA code in the Detectron submodule and in the repo:
```
# ROOT=path/to/cloned/repository
cd $ROOT/Detectron_pytorch/lib
sh make.sh
cd $ROOT/lib
sh make.sh
```

## Annotations

Create a data folder at the top-level directory of the repository:
```
# ROOT=path/to/cloned/repository
cd $ROOT
mkdir data
```
If necessary, one may edit the `DATA_DIR` field in lib/core/config.py to change the expected path to the data directory.

### Visual Genome
Download it [here](https://drive.google.com/open?id=1VDuba95vIPVhg5DiriPtwuVA6mleYGad). Unzip it under the data folder. You should see a `vg` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.

## Images

### Visual Genome
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/vg
mkdir VG_100K
```
Download Visual Genome images from the [official page](https://visualgenome.org/api/v0/api_home.html). Unzip all images (part 1 and part 2) into `VG_100K/`. There should be a total of 108249 files.


## Pre-trained Object Detection Models
Download pre-trained object detection models [here](https://drive.google.com/open?id=1NrqOLbMa_RwHbG3KIXJFWLnlND2kiIpj). Unzip it under the root directory.


## Directory Structure
The final directories for data and detection models should look like:
```
|-- detection_models
|   |-- vg
|   |   |-- VGG16
|   |   |   |-- model_step479999.pth
|-- data
|   |-- vg
|   |   |-- VG_100K    <-- (contains Visual Genome all images)
|   |   |-- rel_annotations_train.json
|   |   |-- rel_annotations_val.json
|   |   |-- ...
|-- trained_models
|   |-- vg_VGG16
|   |   |-- model_step62722.pth
```

## Evaluating Pre-trained Relationship Detection models

DO NOT CHANGE anything in the provided config files(configs/xx/xxxx.yaml) even if you want to test with less or more than 8 GPUs. Use the environment variable `CUDA_VISIBLE_DEVICES` to control how many and which GPUs to use. Remove the
`--multi-gpu-test` for single-gpu inference.

### Visual Genome
**NOTE:** May require at least 64GB RAM to evaluate on the Visual Genome test set

We use three evaluation metrics for Visual Genome:
1. SGCLS: predict subject, object and predicate labels given ground truth subject and object boxes
1. PRDCLS: predict predicate labels given ground truth subject and object boxes and labels

Use `--use_gt_boxes` option to test it with "SGCLS"; use `--use_gt_boxes --use_gt_labels` options to test it with "PRDCLS". The results will vary slightly with the last line of Table 1 in the paper.

To test a trained model using a vg_VGG16 backbone with "SGCLS", run
```
python ./tools/test_net_rel_mps.py --dataset vg --cfg configs/vg/e2e_faster_rcnn_VGG16_8_epochs_vg_v3_default_sgcls.yaml --load_ckpt trained_models/vg_VGG16/model_step62722.pth --output_dir Outputs/vg_VGG16 --multi-gpu-testing --do_val --use_gt_boxes
```
Use `--use_gt_boxes` option to test it with "SGCLS"; use `--use_gt_boxes --use_gt_labels` options to test it with "PRDCLS". The results will vary slightly with those at the last line of Table 1 in the paper.

## Training Relationship Detection Models

The section provides the command-line arguments to train our relationship detection models given the pre-trained object detection models described above. 

DO NOT CHANGE anything in the provided config files(configs/xx/xxxx.yaml) even if you want to train with less or more than 6 GPUs. Use the environment variable `CUDA_VISIBLE_DEVICES` to control how many and which GPUs to use.

With the following command lines, the training results (models and logs) should be in `$ROOT/Outputs/xxx/` where `xxx` is the .yaml file name used in the command without the ".yaml" extension. If you want to test with your trained models, simply run the test commands described above by setting `--load_ckpt` as the path of your trained models.

### Visual Genome
To train our relationship network using a VGG16 backbone, run
```
python tools/train_net_step_rel_mps.py --dataset vg --cfg configs/vg/e2e_faster_rcnn_VGG16_8_epochs_vg_v3_default_sgcls.yaml --nw 6 --use_tfboard
```

## Acknowledgements
This repository uses code based on the [RelDN](https://github.com/NVIDIA/ContrastiveLosses4VRD), as well as
code from the [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) repository by Roy Tseng. See LICENSES for additional details.

## Citing
If you use this code in your research, please use the following BibTeX entry.
```
@misc{lin2020gpsnet,
    title={GPS-Net: Graph Property Sensing Network for Scene Graph Generation},
    author={Xin Lin and Changxing Ding and Jinquan Zeng and Dacheng Tao},
    year={2020},
    eprint={2003.12962},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
