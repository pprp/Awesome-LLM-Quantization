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
| [![Star](https://img.shields.io/github/stars/IST-DASLab/gptq.svg?style=social&label=Star)](https://github.com/IST-DASLab/gptq)[![Publish](https://img.shields.io/badge/Conference-ICLR22-blue)]()<br>[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) <br> Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh <br> [Github](https://github.com/IST-DASLab/gptq) | ![image-20240529203728325](asset/image-20240529203728325.png) | This Paper is the first one to apply post-training quantization to GPT. GPTQ is a one-shot weight quantization method based on approximate second-order information(Hessian). The bit-width is reduced to 3-4 bits per weight. Extreme experiments on 2-bit and ternary quantization are also provided. |
| [![Star](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social&label=Star)]()[![Publish](https://img.shields.io/badge/Conference-ICLR22-blue)]()<br>[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) <br> Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, **Song Han** <br> [Github](https://github.com/mit-han-lab/smoothquant) | ![image-20240529204845286](asset/image-20240529204845286.png) | SmoothQuant is a post-training quantization framework targeting W8A8. In General, |
|                                                              |                                                              |                                                              |






## Contributing

Contributions to this repository are welcome! If you have any suggestions for new resources, or if you find any broken links or outdated information, please open an issue or submit a pull request.

## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT).