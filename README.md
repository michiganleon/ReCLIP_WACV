# ReCLIP: Refine Contrastive Language Image Pre-Training with Source Free Domain Adaptation

## Overview

This repository provides the official PyTorch implementation of our WACV 2024 (Oral) Paper [ReCLIP: Refine Contrastive Language Image Pre-Training with Source Free Domain Adaptation](https://arxiv.org/abs/2308.03793)

### Hardware
We have evaluated our code on NVIDIA A100 GPU with 40GB GPU Memory with batch size of 64. Please use --parallel and smaller batch size for smaller memory GPU.

### Environment
We tested our code with PyTorch 1.12.0.

### Model Weight
We use [CLIP](https://github.com/openai/CLIP) ViT-L/14 as our main base model for adaptation. It is also possible to use other architecture by configing the --architecture option. Our code will automatically download the CLIP checkpoint from [link](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) and put it under the ./ckpt folder. 

### License
ReCLIP is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

<!-- Copyright (c) Amazon. All rights reserved. Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. -->

## Citations

    @article{xuefeng2023reclip,
      title={ReCLIP: Refine Contrastive Language Image Pre-Training with Source Free Domain Adaptation},
      author={Xuefeng, Hu and Ke, Zhang and Lu, Xia and Albert, Chen and Jiajia, Luo and Yuyin, Sun and Ken, Wang and Nan, Qiao and Xiao, Zeng and Min, Sun and others},
      journal={2024 IEEE winter conference on applications of computer vision (WACV)},
      year={2024},
      organization={IEEE}
    }

## Acknowledgements
This work is completed during Xuefeng's internship at Amazon. 
