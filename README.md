

# GraspNet-Edge

GraspNet-Edge is an optimized and efficient 6D grasp pose predictor based on GraspNet, designed for edge devices. 

During the deployment of GraspNet to edge platforms, we observed that many operators were missing when exporting the model to ONNX format. After root cause analysis, we identified that the missing operators originated from CUDA-defined custom layers in PointNet++ (pointnet2). 

To address this issue, we reimplemented these CUDA-based operators using native PyTorch operations and provided corresponding code to enable successful ONNX export.

In addition, we provide parallel training code based on the DDP (Distributed Data Parallel) framework.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/786739982/GraspNet-Edge/">
    <img src="assets/logo.png" alt="Logo" width="146" height="64">
  </a>

  <h3 align="center">GraspNet-Edge</h3>
  <p align="center">
    Optimized GraspNet For Edge Devices !
    <br />
    <a href="https://github.com/786739982/GraspNet-Edge"><strong>Explore the documentation of this project »</strong></a>
    <br />
    <br />
    <a href="https://github.com/786739982/GraspNet-Edge">Demo</a>
    ·
    <a href="https://github.com/786739982/GraspNet-Edge/issues">Report Bug</a>
    ·
    <a href="https://github.com/786739982/GraspNet-Edge/issues">Propose New Feature</a>
  </p>
</p>




## 目录

- [Explanation](#Explanation)
  - [Compare](#Compare)
  - [Implementation](#Implementation)
- [Author](#Author)
- [Acknowledgements](#Acknowledgements)




### Explanation

#### Compare

##### Original GraspNet ONNX Model
<p align="center">
  <a href="https://github.com/786739982/GraspNet-Edge/">
    <img src="assets/GraspNet ONNX.png" alt="Logo" width="" height="">
  </a>
</p>

##### Our GraspNet-Edge ONNX Model (Partial)
<p align="center">
  <a href="https://github.com/786739982/GraspNet-Edge/">
    <img src="assets/GraspNet-Edge ONNX.png" alt="Logo" width="" height="">
  </a>
</p>

#### **Implementation**
* Wrote the code in the ```pointnet2/pointnet2_modules_pytorch.py``` file, which contains operators implemented purely in PyTorch.
* Modified the model code in ```models/backbone.py```, ```models/graspnet.py``` and ```models/modules.py```. Modifications to these files include replacing CUDA-defined operators with PyTorch-defined ones, as well as converting 1D convolutions into equivalent 2D convolutions to better adapt to the hardware characteristics of the Horizon X5 RDK edge device.
* You can refer to the ```train_distributed.py``` file to see how to use DDP to train the model.



### Author

Hongrui Zhu 

E-Mail：786739982@qq.com or hongrui0226@gmail.com

qq:786739982

vx：Hong_Rui_0226



  
### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE](https://github.com/786739982/GraspNet-Edge/blob/master/LICENSE)





### Acknowledgements

- [DISCOVERSE](https://airbots.online/)




<!-- links -->
[contributors-shield]: https://img.shields.io/github/contributors/786739982/GraspNet-Edge.svg?style=flat-square
[contributors-url]: https://github.com/786739982/GraspNet-Edge/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/786739982/GraspNet-Edge.svg?style=flat-square
[forks-url]: https://github.com/786739982/GraspNet-Edge/network/members
[stars-shield]: https://img.shields.io/github/stars/786739982/GraspNet-Edge.svg?style=flat-square
[stars-url]: https://github.com/786739982/GraspNet-Edge/stargazers
[issues-shield]: https://img.shields.io/github/issues/786739982/GraspNet-Edge.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/786739982/GraspNet-Edge.svg
[license-shield]: https://img.shields.io/github/license/786739982/GraspNet-Edge.svg?style=flat-square
[license-url]: https://github.com/786739982/GraspNet-Edge/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555





