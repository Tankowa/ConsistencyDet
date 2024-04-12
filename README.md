# ConsistencyDet: Robust Object Detector with Denoising Paradigm of Consistency Model
Lifan Jiang,Zhihui Wang*,Changmiao Wang,Ming Li,Jiaxu Leng,Xindong Wu

## 1.Network Structure and Detection Results
### 1.1 Structure of the ConsistencyDet

<img src="graphs/structure.png" width="600" height="300"/>

### 1.2 The self-consistency of ConsistencyDet

<img src="graphs/structure2.png" width="450" height="225"/>

### 1.3 Partial Detection Results on the COCO Dataset

<img src="graphs/visualization.png" width="600" height="300"/>

## 2.Abstrat

Object detection, a quintessential task in the realm of perceptual computing, can be tackled using a generative methodology. In the present study, we introduce a novel framework designed to articulate object detection as a denoising diffusion process, which operates on perturbed bounding boxes of annotated entities. This framework, termed ConsistencyDet, leverages an innovative denoising concept known as the Consistency Model. The hallmark of this model is its self-consistency feature, which empowers the model to map distorted information from any temporal stage back to its pristine state, thereby realizing a ``one-step denoising'' mechanism. Such an attribute markedly elevates the operational efficiency of the model, setting it apart from the conventional Diffusion Model. Throughout the training phase, ConsistencyDet initiates the diffusion sequence with noise-infused boxes derived from the ground-truth annotations and conditions the model to perform the denoising task. Subsequently, in the inference stage, the model employs a denoising sampling strategy that commences with bounding boxes randomly sampled from a normal distribution. Through iterative refinement, the model transforms an assortment of arbitrarily generated boxes into the definitive detections. Comprehensive evaluations employing standard benchmarks, such as MS-COCO and LVIS, corroborate that ConsistencyDet surpasses other leading-edge detectors in performance metrics. 


## 3.contributes 

<ul>
    <li>
        <h3>One</h3>
        <p>We conceptualize object detection as a generative denoising process and propose a novel methodological approach. In contrast to the established paradigm in DiffusionDet, which employs an equal number of iterations for noise addition and removal, our method represents a substantial advancement in enhancing the efficiency of the detection task.</p>
    </li>
    <li>
        <h3>Two</h3>
        <p>In the proposed ConsistencyDet, we have engineered a noise addition and removal paradigm that does not impose specific architectural constraints, thereby allowing for flexible parameterization with a variety of neural network structures. This design choice significantly augments the model's practicality and adaptability for diverse applications.</p>
    </li>
    <li>
        <h3>Three</h3>
        <p>In crafting the loss function for the proposed ConsistencyDet, we aggregate the individual loss values at time steps $t$ and $t-1$ subsequent to the model's predictions to compute the total loss. This methodology guarantees that the mapping of any pair of adjacent points along the temporal dimension to the axis origin maintains the highest degree of consistency. This attribute mirrors the inherent self-consistency principle central to the Consistency Model.</p>
    </li>
</ul>

## 4.Experimental results
<table border="1">
  <tr>
    <th>Method</th>
    <th>The best Box AP</th>
    <th>Download</th>
  </tr>
  <tr>
    <td>COCO-Res50</td>
    <td>46.9</td>
    <td><a href="https://pan.baidu.com/s/14GFs5oBZeV6XWk6xiNDmXg?pwd=1111">model</a></td>
  </tr>
  <tr>
    <td>COCO-Res101</td>
    <td>47.2</td>
    <td><a href="https://pan.baidu.com/s/1Rj7TMGt1cOBubkRutypObA?pwd=1111" download>model</a></td>
  </tr>
  <tr>
    <td>COCO-SwinBase</td>
    <td>53.0</td>
    <td><a href="https://pan.baidu.com/s/1zgfJip_HSx0FAB4EiyX8tA?pwd=1111" download>model</a></td>
  </tr>
  <tr>
    <td>LVIS-Res50</td>
    <td>32.2</td>
    <td><a href="https://pan.baidu.com/s/19ELAf3xNf6uYtILmyFFmvQ?pwd=1111" download>model</a></td>
  </tr>
  <tr>
    <td>LVIS-Res101</td>
    <td>33.1</td>
    <td><a href="https://pan.baidu.com/s/1wXPChzSKMVRHiB7DYDsG3Q?pwd=1111" download>model</a></td>
  </tr>
  <tr>
    <td>LVIS-SwinBase</td>
    <td>42.4</td>
    <td><a href="https://pan.baidu.com/s/1KpT-3ktSYM_R1n5hBn8Nsw?pwd=1111" download>model</a></td>
  </tr>
</table>

## 5.Installation
1.Install anaconda, and create conda environment;
<pre>
conda create -n yourname python=3.8
</pre>
2.PyTorch ≥ 1.9.0 and torchvision that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
3.Install Detectron2
<pre>
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
</pre>
4.Install other dependency libraries
<pre>
pip3 install -r requirements.txt
</pre>

## 6.Data preparation
<pre>
mkdir -p datasets/coco
mkdir -p datasets/lvis

ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017

ln -s /path_to_lvis_dataset/lvis_v1_train.json datasets/lvis/lvis_v1_train.json
ln -s /path_to_lvis_dataset/lvis_v1_val.json datasets/lvis/lvis_v1_val.json
</pre>

## 7.Prepare pretrain models
<pre>
mkdir models
cd models
# ResNet-101
wget https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/torchvision-R-101.pkl

# Swin-Base
wget https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/swin_base_patch4_window7_224_22k.pkl
</pre>

## 8.Training
<pre>
python train_net.py --num-gpus 4 \
  --config-file configs/diffdet.coco.res50.yaml
</pre>

## 9.Evaling
<pre>
python train_net.py --num-gpus 4 \
  --config-file configs/diffdet.yourdataset.yourbakbone.yaml \
  --eval-only MODEL.WEIGHTS path/to/model.pth
</pre>

## 10.Citation
@misc{jiang2024consistencydet,
      title={ConsistencyDet: Robust Object Detector with Denoising Paradigm of Consistency Model}, 
      author={Lifan Jiang and Zhihui Wang and Changmiao Wang and Ming Li and Jiaxu Leng and Xindong Wu},
      year={2024},
      eprint={2404.07773},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

## 11.Acknowledgement
A large part of the code is borrowed from DiffusionDet and Consistency models thanks for their works.
<pre>
@inproceedings{chen2023diffusiondet,
  title={Diffusiondet: Diffusion model for object detection},
  author={Chen, Shoufa and Sun, Peize and Song, Yibing and Luo, Ping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19830--19843},
  year={2023}
}
@article{song2023consistency,
  title={Consistency models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023}
}
</pre>

