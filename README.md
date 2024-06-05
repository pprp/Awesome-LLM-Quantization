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
| ACL24 Long Paper<br />[BitDistiller: Unleashing the Potential of Sub 4-bit LLMs via Self-Distillation](http://arxiv.org/abs/2402.10631)<br />Dayou Du, Yijia Zhang, Shijie Cao, Jiaqi Guo, Ting Cao, Xiaowen Chu, Ningyi Xu<br />[Github](https://github.com/DD-DuDa/BitDistiller) | ![image-20240603211630312](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240603211630312.png) | Bitdistiller is a QAT framework that utilizes Knowledge Distillation to boost the performance at Sub-4bit. BitDistiller (1) incorporates a tailored asymmetric quantization and clipping technique to perserve the fidelity of quantized weight and (2) proposes a Confidence-Aware Kullback-Leibler Divergence (CAKLD) as self-distillation loss. Experiments involve 3-bit and 2-bit configuration. <br />#QAT #2-bit #3-bit #KD |
| ICML24 <br>[BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291) <br> Wei Huang, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, Xiaojuan Qi <br> [Github](https://github.com/Aaronhuang-778/BiLLM) | ![image-20240529225417121](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240529225417121.png) | BiLLM is the first 1-bit post-training quatization framework for pretrained LLMs. BiLLM split the weights into salient weight and non-salient one. For the salient weights, they propose the binary residual approximation strategy. For the unsalient weights, they propose an optimal splitting search to group and binarize them independently. BiLLM achieve 8.41 ppl on LLaMA2-70B with only 1.08 bit.<br />#PTQ #1-bit |
| Arxiv24 <br>[SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models](https://arxiv.org/abs/2405.14917) <br> Wei Huang, Haotong Qin, Yangdong Liu, Yawei Li, Xianglong Liu, Luca Benini, Michele Magno, Xiaojuan Qi <br> [Github](https://github.com/Aaronhuang-778/SliM-LLM) | ![image-20240529231050896](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240529231050896.png) | This paper presents Slience-Driven Mixed-Precision Quantization for LLMs, called Slim-LLM, targeting 2-bit mixed precision quantization. Specifically, Silm-LLM involves two techniques: (1) Salience-Determined Bit Allocation (SBA): by minimizing the KL divergence between original output and the quantized output, the objective is to find the best bit assignment for each group. (2) Salience-Weighted Quantizer Calibration: by considering the element-wise salience within the group, Slim-LLM search for the calibration parameter $\gamma$ to prevent the degradation of local salient weight information in each group.<br />#MixedPrecision #2-bit |
| Arxiv24<br>[AdpQ:A Zero-shot Calibration Free Adaptive Post Training Quantization Method for LLMs](https://arxiv.org/abs/2405.13358v1) | ![image-20240603205727796](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240603205727796.png) | This paper formulates outlier weight identification problem in PTQ as the concept of shinkage in statistical ML. By applying Adaptive LASSO regression model, AdpQ ensures the quantized weights distirbution is close to that of origin, thus eliminating the requirement of calibration data. Lasso Regression employ the L1 regularization and minimize the KL divergence between the original weight and quantized one. The experiments mainly focus on 3/4 bit quantization<br />#PTQ #Regression |
| Arxiv24<br /> [Integer Scale: A Free Lunch for Faster Fine-grained Quantization for LLMs](https://arxiv.org/abs/2405.14597v2) <br /> Qingyuan Li, Ran Meng, Yiduo Li, Bo Zhang, Yifan Lu, Yerui Sun, Lin Ma, Yuchen Xie | ![image-20240603212845404](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240603212845404.png) | Integer Scale is a PTQ framework that require no extra calibration and maintain the performance. It is a fine-grained quantization method and can be used plug-and-play. Specifically, it reorder the sequence of costly type conversion I32toF32. <br />#PTQ #W4A16 #W4A8 |
| Arxiv24<br />[QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)<br />Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman<br />[Github](https://github.com/spcl/QuaRot) | ![image-20240603215015646](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240603215015646.png) | This paper introduce a novel rotation-based quantization scheme, which can quantize the weight, activation, and KV cache of LLMs in 4-bit. QuaRot rotates the LLMs to remove the outliers from hideenstate. It apply randomized Hadamard transformations to the weight matrices without changing the model. When applying this transformation to attention module, it enables the KV cache quantization.  <br />#PTQ #4bit #Rotation |
| Arxiv24<br />[SpinQuant:LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406v2)<br />Zechun Liu, Changsheng Zhao etc. | ![image-20240603214042270](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240603214042270.png) | Rotating activation or weight matrices heps remove outliers and benefits quantizaion (rotational invariance property). They first identify a collection of applicable rotation parameterizations that lead to identical outputs in full-precision Transformer. They find that soem **random rotations lead to better quantization** than others. Then, SpinQuant was proposed to optimize the rotation matrices with *Cayley* optimization on validation dataset. Specifically, them employ Cayley SGD method to optimize the rotation matrix on the Stiefel manifold. <br />#PTQ #Rotation #4bit |
| NeurIPS2024 <br /> [Outliers and Calibration Sets have Dimishing Effect on Quantization of Mordern LLMs](https://arxiv.org/abs/2405.20835v1)<br /> Davide Paglieri, Saurabh Dash, Tim Rocktaschel, Jack Parker-Holder | ![image-20240604110208446](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240604110208446.png) | This paper evaluates the effects of calibration set on the performance of LLM quantization, especially on hidden activations. Calibration set can distort the quantization range and negatively impact performance. This paper reveals that different model has shown different tendency towards quantization. (1) OPT has shown high susceptibility to outliers with varying calibration sets. (2) Newer models like Llama-2-7B, Llama-3-8B, Mistral-7B has demonstrated stronger robustness. This findings suggest a shift in PTQ strategies. These findings indicate that we should emphasis more on optimizing inference speed rather than focusing on outlier preservation. <br />#Analysis #Evaluation #Finding |
| Arxiv24<br /> [Effective Interplay between Sparsity and Quantization: From Theory to Practice]()<br /> Simla Burcu Harma Ayan Chakraborty, Elizaveta Kostenok, Dnila Mishin, etc. <br /> [Github](https://github.com/parsa-epfl/quantization-sparsity-interplay) | ![image-20240604125632861](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240604125632861.png) | This paper dives into the interplay between sparsity and quantization and evaluates whether thheir combination impacts final performance of LLMs. This paper theriotically proves that applying sparsity before quantization is the optimal sequence, minimizing the error in computation. The experiments involves OPT, LLaMA and ViT. Findings: (1) sparsity and quantization are not orthogonal; (2) interaction between Sparsity and quantization significantly harm the performance, where quantization error is playing a dominant role in the degradation. <br />#Theory #Sparisty |
| NeurIPS23<br /> [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)<br /> Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer <br /> [Github](https://github.com/artidoro/qlora) | ![image-20240604191811055](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240604191811055.png) | QLoRA aims to reduce the memory usage of LLM by incoorporating the LoRA with 4-bit quantized pretrained model. Specifically, QLoRA introduces (1) 4-bit NormalFlot(NF4), a information theoretically optimal for  normally distributed weights. (2) double quantization to reduce the memory footprint. (3) paged optimizers to manage memory spikes. <br />#NF4 #4-bit #LoRA |
| NeurIPS23<br />[QuIP: 2-Bit Quantization of Large Language Models with Guarantees](https://proceedings.neurips.cc/paper_files/paper/2023/file/0df38cd13520747e1e64e5b123a78ef8-Paper-Conference.pdf) <br />Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa <br /> [Github](https://github.com/Cornell-RelaxML/QuIP) |                                                              | This paper introduces Quantization with Incoherence Processing (QuIP), which is based on the insight that quantization benefits from incoherent weight and Hessian matrices. It consists of two steps (1) adaptive rounding procedure to minimize a quadratic proxy objective. (2) pre- and post-processing that ensures weight and Hessian incoherence using random orthgonal matrices. QuIP makes the two-bit LLM compression viable for the first time. <br />#PTQ #2-bit #Rotation |
| ICLR24 Spotlight<br /> [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/pdf/2308.13137) <br /> Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo <br /> [Github](https://github.com/OpenGVLab/OmniQuant) | ![image-20240605153427027](https://github.com/pprp/Awesome-LLM-Quantization/blob/main/asset/image-20240605153427027.png) |                                                              |



## Perplexity Evaluation

**LLaMA 1 - FP16 - Baseline (Perplexity on Wikitext2**

| Model                   | Quantization Technique | 1-7B | 1-13B | 1-30B | 1-65B | 2-7B | 2-13B | 2-70B |
| :---------------------- | :--------------------- | :--- | :---- | :---- | :---- | :--- | :---- | :---- |
| LLaMA 1 - FP16 - W16A16 | -                      | 5.68 | 5.09  | 4.10  | 3.53  | 5.47 | 4.88  | 3.31  |

**LLaMA 2 - W2A16**

| Quantization Technique               | 1-7B      | 1-13B     | 1-30B    | 1-65B    | 2-7B      | 2-13B     | 2-70B    |
| :----------------------------------- | :-------- | :-------- | :------- | :------- | :-------- | :-------- | :------- |
| Round-to-Nearest                     | 1.1e5     | 6.8e4     | 2.4e4    | 2.2e4    | 3.8e4     | 5.6e4     | 2.0e4    |
| Generative Pre-Training Quantization | 2.1e3     | 5.5e3     | 499.75   | 55.91    | 7.7e3     | 2.1e3     | 77.95    |
| **OmniQuant**                        | **15.47** | **13.21** | **8.71** | **7.58** | **37.37** | **17.21** | **7.81** |

**LLaMA 2 - W2A16 g128**

| Quantization Technique               | 1-7B     | 1-13B    | 1-30B    | 1-65B    | 2-7B      | 2-13B    | 2-70B    |
| :----------------------------------- | :------- | :------- | :------- | :------- | :-------- | :------- | :------- |
| Round-to-Nearest                     | 1.9e3    | 781.20   | 68.04    | 15.08    | 4.2e3     | 122.08   | 27.27    |
| Generative Pre-Training Quantization | 44.01    | 15.60    | 10.92    | 9.51     | 36.77     | 28.14    | NAN      |
| AWQ                                  | 2.6e5    | 2.8e5    | 2.4e5    | 7.4e4    | 2.2e5     | 1.2e5    | -        |
| **OmniQuant**                        | **9.72** | **7.93** | **7.12** | **5.95** | **11.06** | **8.26** | **6.55** |

**LLaMA 2 - W2A16 g64**

| Quantization Technique               | 1-7B     | 1-13B    | 1-30B    | 1-65B    | 2-7B     | 2-13B    | 2-70B    |
| :----------------------------------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| Round-to-Nearest                     | 188.32   | 101.87   | 19.20    | 9.39     | 431.97   | 26.22    | 10.31    |
| Generative Pre-Training Quantization | 22.10    | 10.06    | 8.54     | 8.31     | 20.85    | 22.44    | NAN      |
| AWQ                                  | 2.5e5    | 2.7e5    | 2.3e5    | 7.4e4    | 2.1e5    | 1.2e5    | -        |
| **OmniQuant**                        | **8.90** | **7.34** | **6.59** | **5.65** | **9.62** | **7.56** | **6.11** |

**LLaMA 2 - W3A16**

| Quantization Technique               | 1-7B     | 1-13B    | 1-30B    | 1-65B    | 2-7B     | 2-13B    | 2-70B    |
| :----------------------------------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| Round-to-Nearest                     | 25.73    | 11.39    | 14.95    | 10.68    | 539.48   | 10.68    | 7.52     |
| Generative Pre-Training Quantization | 8.06     | 6.76     | 5.84     | 5.06     | 8.37     | 6.44     | 4.82     |
| AWQ                                  | 11.88    | 7.45     | 10.07    | 5.21     | 24.00    | 10.45    | -        |
| **OmniQuant**                        | **6.49** | **5.68** | **4.74** | **4.04** | **6.58** | **5.58** | **3.92** |

**LLaMA 2 - W3A16 g128**

| Quantization Technique               | 1-7B     | 1-13B    | 1-30B    | 1-65B    | 2-7B     | 2-13B    | 2-70B    |
| :----------------------------------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| Round-to-Nearest                     | 7.01     | 5.88     | 4.87     | 4.24     | 6.66     | 5.51     | 3.97     |
| Generative Pre-Training Quantization | 6.55     | 5.62     | 4.80     | 4.17     | 6.29     | 5.42     | 3.85     |
| AWQ                                  | 6.46     | 5.51     | 4.63     | 3.99     | 6.24     | 5.32     | -        |
| **OmniQuant**                        | **6.15** | **5.44** | **4.56** | **3.94** | **6.03** | **5.28** | **3.78** |

**LLaMA 2 - W4A16**

| Quantization Technique               | 1-7B     | 1-13B    | 1-30B    | 1-65B    | 2-7B     | 2-13B    | 2-70B    |
| :----------------------------------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| Round-to-Nearest                     | 6.43     | 5.55     | 4.57     | 3.87     | 6.11     | 5.20     | 3.67     |
| Generative Pre-Training Quantization | 6.13     | 5.40     | 4.48     | 3.83     | 5.83     | 5.13     | 3.58     |
| AWQ                                  | 6.08     | 5.34     | 4.39     | 3.76     | 6.15     | 5.12     | -        |
| **OmniQuant**                        | **5.86** | **5.21** | **4.25** | **3.71** | **5.74** | **5.02** | **3.47** |

**LLaMA 2 - W4A16 g128**

| Quantization Technique               | 1-7B     | 1-13B    | 1-30B    | 1-65B    | 2-7B     | 2-13B    | 2-70B    |
| :----------------------------------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| Round-to-Nearest                     | 5.96     | 5.25     | 4.23     | 3.67     | 5.72     | 4.98     | 3.46     |
| Generative Pre-Training Quantization | 5.85     | 5.20     | 4.23     | 3.65     | 5.61     | 4.98     | 3.42     |
| AWQ                                  | 5.81     | 5.20     | 4.21     | 3.62     | 5.62     | 4.97     | -        |
| **OmniQuant**                        | **5.77** | **5.17** | **4.19** | **3.62** | **5.58** | **4.95** | **3.40** |



## Contributing

Contributions to this repository are welcome! If you have any suggestions for new resources, or if you find any broken links or outdated information, please open an issue or submit a pull request.


## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT).
