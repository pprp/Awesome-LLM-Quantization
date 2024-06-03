# Awesome-LLM-Quantization

<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/stars/pprp/Awesome-LLM-Quantization.svg?style=social >
  <img src=https://img.shields.io/github/watchers/pprp/Awesome-LLM-Quantization.svg?style=social >
  <img src=https://img.shields.io/badge/Release-v0.1-brightgreen.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>   

Welcome to the Awesome-LLM-Quantization repository! This is a curated list of resources related to quantization techniques for Large Language Models (LLMs). Quantization is a crucial step in deploying LLMs on resource-constrained devices, such as mobile phones or edge devices, by reducing the model's size and computational requirements.

## Contents

This repository contains the following sections:

- **Papers**: A collection of research papers and articles on quantization techniques for LLMs.
- **Tutorials**: Step-by-step guides and tutorials for implementing quantization methods.
- **Libraries**: A list of open-source libraries and frameworks that support quantization for LLMs.
- **Datasets**: Datasets commonly used for benchmarking and evaluating quantized LLMs.
- **Tools**: Useful tools and utilities for quantization and deployment of LLMs.


## Papers 

| Title & Author & Link                                        | Introduction                                                 | Summary                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ICLR22 <br>[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) <br> Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh <br> [Github](https://github.com/IST-DASLab/gptq) | ![image-20240529203728325](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240529203728325.png) | This Paper is the first one to apply post-training quantization to GPT. GPTQ is a one-shot weight quantization method based on approximate second-order information(Hessian). The bit-width is reduced to 3-4 bits per weight. Extreme experiments on 2-bit and ternary quantization are also provided. <br />#PTQ #3-bit #4-bit #2-bit |
| ICML23<br>[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) <br> Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, **Song Han** <br> [Github](https://github.com/mit-han-lab/smoothquant) | ![image-20240529213948826](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240529213948826.png) | SmoothQuant is a post-training quantization framework targeting W8A8 (INT8). In General, weights are easier to quantize than activation. It propose to migrate the quantization difficulty from activations to weights using mathematically equivalent transformation using $`s = \frac{\left( \max \left( \lvert X \rvert \right) \right)^\alpha}{\left( \max \left( \lvert W \rvert \right) \right)^{1-\alpha}}`$.<br />#PTQ #W8A8 |
| MLSys24_BestPaper <br> [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) <br>Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han, **Song Han** <br> [Github](https://github.com/mit-han-lab/llm-awq) | ![image-20240529214839469](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240529214839469.png) | Activation-aware Weight Quantization (AWQ) is low-bit weight-only quantization method targeting edge devices with W4A16. The motivation is protecting only 1% of sliant weighs can retain the performance. Then, AWQ aims to search for the optimal per-channel scaling $`s^* = \arg\min_{s} \left\Vert Q \left( W \cdot \text{diag}(s) \right) \left( \text{diag}(s)^{-1} \cdot X \right) - W X \right\Vert`$ to protect the salient weights by observing the activation. Then, we have $Q(w\cdot s)\cdot \frac{x}{s}$.<br />#PTQ #W4A16 |
| ACL24 Long Paper<br />[BitDistiller: Unleashing the Potential of Sub 4-bit LLMs via Self-Distillation](http://arxiv.org/abs/2402.10631)<br />Dayou Du, Yijia Zhang, Shijie Cao, Jiaqi Guo, Ting Cao, Xiaowen Chu, Ningyi Xu<br />[Github](https://github.com/DD-DuDa/BitDistiller) | ![image-20240603211630312](asset/image-20240603211630312.png) | Bitdistiller is a QAT framework that utilizes Knowledge Distillation to boost the performance at Sub-4bit. BitDistiller (1) incorporates a tailored asymmetric quantization and clipping technique to perserve the fidelity of quantized weight and (2) proposes a Confidence-Aware Kullback-Leibler Divergence (CAKLD) as self-distillation loss. Experiments involve 3-bit and 2-bit configuration. <br />#QAT #2-bit #3-bit #KD |
| ICML24 <br>[BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291) <br> Wei Huang, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, Xiaojuan Qi <br> [Github](https://github.com/Aaronhuang-778/BiLLM) | ![image-20240529225417121](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240529225417121.png) | BiLLM is the first 1-bit post-training quatization framework for pretrained LLMs. BiLLM split the weights into salient weight and non-salient one. For the salient weights, they propose the binary residual approximation strategy. For the unsalient weights, they propose an optimal splitting search to group and binarize them independently. BiLLM achieve 8.41 ppl on LLaMA2-70B with only 1.08 bit.<br />#PTQ #1-bit |
| Arxiv24 <br>[SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models](https://arxiv.org/abs/2405.14917) <br> Wei Huang, Haotong Qin, Yangdong Liu, Yawei Li, Xianglong Liu, Luca Benini, Michele Magno, Xiaojuan Qi <br> [Github](https://github.com/Aaronhuang-778/SliM-LLM) | ![image-20240529231050896](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240529231050896.png) | This paper presents Slience-Driven Mixed-Precision Quantization for LLMs, called Slim-LLM, targeting 2-bit mixed precision quantization. Specifically, Silm-LLM involves two techniques: (1) Salience-Determined Bit Allocation (SBA): by minimizing the KL divergence between original output and the quantized output, the objective is to find the best bit assignment for each group. (2) Salience-Weighted Quantizer Calibration: by considering the element-wise salience within the group, Slim-LLM search for the calibration parameter $\gamma$ to prevent the degradation of local salient weight information in each group.<br />#MixedPrecision #2-bit |
| Arxiv24<br>[AdpQ:A Zero-shot Calibration Free Adaptive Post Training Quantization Method for LLMs](https://arxiv.org/abs/2405.13358v1) | ![image-20240603205727796](asset/image-20240603205727796.png) | This paper formulates outlier weight identification problem in PTQ as the concept of shinkage in statistical ML. By applying Adaptive LASSO regression model, AdpQ ensures the quantized weights distirbution is close to that of origin, thus eliminating the requirement of calibration data. Lasso Regression employ the L1 regularization and minimize the KL divergence between the original weight and quantized one. The experiments mainly focus on 3/4 bit quantization<br />#PTQ #Regression |
|                                                              |                                                              |                                                              |






## Contributing

Contributions to this repository are welcome! If you have any suggestions for new resources, or if you find any broken links or outdated information, please open an issue or submit a pull request.


## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT).
