# Papers for Learning based Video Coding

**Purpose**: We aim to provide a summary of learning based video coding. More papers will be summarized.

University of Science and Technology of China (USTC), [Intelligent Media Computing Lab](https://scholar.google.com/citations?user=1ayDJfsAAAAJ&hl=en&oi=ao)

**📌 About new works.** If you want to incorporate your studies (e.g., the link of paper or project) on diffusion model-based image processing in this repository. Welcome to raise an issue or email us. We will incorporate it into this repository and our survey report as soon as possible. 

## Table of contents
<!-- - [Survey paper](#survey-paper)
- [Table of contents](#table-of-contents) -->
- [Autoencoder based Coding](#autoencoder-based-coding-model)
- [Hybrid Coding](#hybrid-coding-model)
  - [Residual Coding](#residual-coding)
  - [Conditional Coding](#conditional-coding)
- [INR based Coding](#inr-based-coding)

### Autoencoder based Coding Model
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
| VCT | [VCT: A Video Compression Transformer](https://proceedings.neurips.cc/paper_files/paper/2022/file/54dcf25318f9de5a7a01f0a4125c541e-Paper-Conference.pdf) | Fabian Mentzer | NeurIPS 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/google-research/google-research)](https://github.com/google-research/google-research/tree/master/vct) [![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/NeuralCompression)](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/torch_vct) |
|  | [Conditional Entropy Coding for Efficient Video Compression](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620443.pdf) | Jerry Liu | ECCV 2020 |  |
|  | [Video Compression With Rate-Distortion Autoencoders](https://openaccess.thecvf.com/content_ICCV_2019/papers/Habibian_Video_Compression_With_Rate-Distortion_Autoencoders_ICCV_2019_paper.pdf) | Amirhossein Habibian | ICCV 2019 |  |

### Hybrid Coding Model

#### Residual Coding
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
| ENVC | [Learning Cross-Scale Weighted Prediction for Efficient Neural Video Compression](https://ieeexplore.ieee.org/abstract/document/10159648) | Zongyu Guo | TIP 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/USTC-IMCL/ENVC)](https://github.com/USTC-IMCL/ENVC) |
| C2F | [Coarse-to-fine Deep Video Coding with Hyperprior-guided Mode Prediction](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Coarse-To-Fine_Deep_Video_Coding_With_Hyperprior-Guided_Mode_Prediction_CVPR_2022_paper.pdf) | Zhihao Hu | CVPR 2022 |  |
| RLVC | [Learning for Video Compression With Recurrent Auto-Encoder and Recurrent Probability Model](https://ieeexplore.ieee.org/abstract/document/9288876) | Ren Yang | JSTSP 2020 | [![GitHub Repo stars](https://img.shields.io/github/stars/RenYang-home/RLVC)](https://github.com/RenYang-home/RLVC) |
| FVC | [FVC: A New Framework towards Deep Video Compression in Feature Space](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_FVC_A_New_Framework_Towards_Deep_Video_Compression_in_Feature_CVPR_2021_paper.pdf) | Zhihao Hu | CVPR 2021 | [![GitHub Repo stars](https://img.shields.io/github/stars/ZhihaoHu/PyTorchVideoCompression)](https://github.com/ZhihaoHu/PyTorchVideoCompression/tree/master/FVC) |
| DVC-Pro | [An End-to-End Learning Framework for Video Compression](https://ieeexplore.ieee.org/abstract/document/9072487) | Guo Lu | TPAMI 2020 |  |
| M-LVC | [M-LVC: Multiple Frames Prediction for Learned Video Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_M-LVC_Multiple_Frames_Prediction_for_Learned_Video_Compression_CVPR_2020_paper.pdf) | Jianping Lin | CVPR 2020 | [![GitHub Repo stars](https://img.shields.io/github/stars/JianpingLin/M-LVC_CVPR2020)](https://github.com/JianpingLin/M-LVC_CVPR2020) |
| SSF | [Scale-space flow for end-to-end optimized video compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.pdf) | Eirikur Agustsson | CVPR 2020 | [![GitHub Repo stars](https://img.shields.io/github/stars/InterDigitalInc/CompressAI)](https://github.com/InterDigitalInc/CompressAI) |
| RaFC | [Improving Deep Video Compression by Resolution-Adaptive Flow Coding](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470188.pdf) | Zhihao Hu | ECCV 2020 |  |
| DVC | [DVC: An End-to-end Deep Video Compression Framework](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.pdf) | Guo Lu | CVPR 2019 | [![GitHub Repo stars](https://img.shields.io/github/stars/GuoLusjtu/DVC)](https://github.com/GuoLusjtu/DVC) [![GitHub Repo stars](https://img.shields.io/github/stars/ZhihaoHu/PyTorchVideoCompression)](https://github.com/zhihaohu/pytorchvideocompression) [![GitHub Repo stars](https://img.shields.io/github/stars/RenYang-home/OpenDVC)](https://github.com/RenYang-home/OpenDVC) |
|  | [Learned Video Compression](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rippel_Learned_Video_Compression_ICCV_2019_paper.pdf) | Oren Rippel | ICCV 2019 |  |
| PMCNN | [Learning for Video Compression](https://ieeexplore.ieee.org/document/8610323) | Zhibo Chen | TCSVT 2019 |  |

#### Conditional Coding
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
| DCVC-FM | [Neural Video Compression with Feature Modulation](https://arxiv.org/abs/2402.17414) | Jiahao Li | CVPR 2024 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC-FM) |
| DCVC-DC | [Neural Video Compression with Diverse Contexts](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Neural_Video_Compression_With_Diverse_Contexts_CVPR_2023_paper.pdf) | Jiahao Li | CVPR 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC-DC) |
| DCVC-MIP | [Motion Information Propagation for Neural Video Compression](https://openaccess.thecvf.com/content/CVPR2023/papers/Qi_Motion_Information_Propagation_for_Neural_Video_Compression_CVPR_2023_paper.pdf) | Linfeng Qi | CVPR 2023 |  |
| MIMT | [MIMT: Masked Image Modeling Transformer for Video Compression](https://openreview.net/pdf?id=j9m-mVnndbm) | Jinxi Xiang | ICLR 2023 |  |
| DCVC-HEM | [Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression](https://dl.acm.org/doi/pdf/10.1145/3503161.3547845) | Jiahao Li | ACM MM 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC-HEM) |
| CANF-VC | [CANF-VC: Conditional Augmented Normalizing Flows for Video Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760193.pdf) | Yung-Han Ho | ECCV 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/NYCU-MAPL/CANF-VC)](https://github.com/NYCU-MAPL/CANF-VC) |
| DCVC-TCM | [Temporal Context Mining for Learned Video Compression](https://ieeexplore.ieee.org/abstract/document/9941493) | Xihua Sheng | TMM 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC-TCM) |
| DCVC | [Deep Contextual Video Compression](https://proceedings.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf) | Jiahao Li | NeurIPS 2021 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC) |
| ELF-VC | [ELF-VC: Efficient Learned Flexible-Rate Video Coding](https://openaccess.thecvf.com/content/ICCV2021/papers/Rippel_ELF-VC_Efficient_Learned_Flexible-Rate_Video_Coding_ICCV_2021_paper.pdf) | Oren Rippel, Alexander G. Anderson | ICCV 2021 |  |


### INR based Coding
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [Boosting Neural Representations for Videos with a Conditional Decoder](https://arxiv.org/pdf/2402.18152.pdf) | Xinjie Zhang | CVPR 2024 |  |
| HiNeRV | [HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation](https://proceedings.neurips.cc/paper_files/paper/2023/file/e5dc475c370ff42f2f96dddf8191a40c-Paper-Conference.pdf) | Ho Man Kwan | NeurIPS 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/hmkx/HiNeRV)](https://github.com/hmkx/HiNeRV) |
| HNeRV | [HNeRV: A Hybrid Neural Representation for Videos](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.pdf) | Hao Chen | CVPR 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/haochen-rye/HNeRV)](https://github.com/haochen-rye/HNeRV) |
| E-NeRV | [E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950263.pdf) | Zizhang Li | ECCV 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/kyleleey/E-NeRV)](https://github.com/kyleleey/E-NeRV) |
| NeRV | [NeRV: Neural Representations for Videos](https://proceedings.neurips.cc/paper_files/paper/2021/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf) | Hao Chen | NeurIPS 2021 | [![GitHub Repo stars](https://img.shields.io/github/stars/haochen-rye/NeRV)](https://github.com/haochen-rye/NeRV) |
