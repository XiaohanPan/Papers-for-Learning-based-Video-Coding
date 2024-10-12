# Papers for Learning based Video Coding

**Purpose**: We aim to provide a summary of learning based video coding. More papers will be summarized.

University of Science and Technology of China (USTC), [Intelligent Media Computing Lab](https://scholar.google.com/citations?user=1ayDJfsAAAAJ&hl=en&oi=ao)

**📌 About new works.** If you want to incorporate your studies (e.g., the link of paper or project) on diffusion model-based image processing in this repository. Welcome to raise an issue or email us. We will incorporate it into this repository and our survey report as soon as possible. 

## Table of contents
<!-- - [Survey paper](#survey-paper)
- [Table of contents](#table-of-contents) -->
- [Uncategorized Papers](#uncategorized-papers)
- [Codec Performance](#codec-performance)
  - [Autoencoder based Coding](#autoencoder-based-coding-model)
  - [Hybrid Coding](#hybrid-coding-model)
    - [Conditional Residual Coding](#conditional-residual-coding)
    - [Conditional Coding](#conditional-coding)
    - [Residual Coding](#residual-coding)
  - [INR based Coding](#inr-based-coding)
  - [Framework-independent Optimization](#framework-independent-optimization)
- [Codec Scalability](#codec-scalability)
- [Codec Practicality](#codec-practicality)

## Uncategorized Papers
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [SNeRV: Spectra-preserving Neural Representation for Video](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07231.pdf) | Jina Kim* , Jihoo Lee* | ECCV 2024 |  |
|  | [Efficient Neural Video Representation with Temporally Coherent Modulation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09437.pdf) | Seungjun Shin | ECCV 2024 |  |
|  | [Fast Encoding and Decoding for Implicit Video Representation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05618.pdf) | Hao Chen | ECCV 2024 |  |
|  | [High-Efficiency Neural Video Compression via Hierarchical Predictive Learning](https://arxiv.org/pdf/2410.02598) | Ming Lu | arxiv 2024.10 |  |
|  | [Releasing the Parameter Latency of Neural Representation for High-Efficiency Video Compression](https://arxiv.org/pdf/2410.01654) | Gai Zhang | arxiv 2024.10 |  |
|  | [Learned Rate Control for Frame-Level Adaptive Neural Video Compression via Dynamic Neural Network](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11394.pdf) | Chenhao Zhang | ECCV 2024 |  |
|  | [Free-VSC: Free Semantics from Visual Foundation Models for Unsupervised Video Semantic Compression](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06543.pdf) | Yuan Tian | ECCV 2024 |  |
|  | [All-in-One Image Coding for Joint Human-Machine Vision with Multi-Path Aggregation](https://arxiv.org/pdf/2409.19660) | Xu Zhang | arxiv 2024.09 |  |
|  | [NVRC: Neural Video Representation Compression](https://arxiv.org/pdf/2409.07414) | Ho Man Kwan | arxiv 2024.09 |  |
|  | [USTC-TD: A Test Dataset and Benchmark for Image and Video Coding in 2020s](https://arxiv.org/pdf/2409.08481) | Zhuoyuan Li*, Junqi Liao* | arxiv 2024.09 |  |
|  | [PNVC: Towards Practical INR-based Video Compression](https://arxiv.org/pdf/2409.00953) | Ge Gao | arXiv 2024.09 |  |
|  | [I2VC: A Unified Framework for Intra- & Inter-frame Video Compression](https://arxiv.org/pdf/2405.14336) | Meiqin Liu | arXiv 2024.05 | [![GitHub Repo stars](https://img.shields.io/github/stars/GYukai/I2VC)](https://github.com/GYukai/I2VC) |
|  | [MambaVC: Learned Visual Compression with Selective State Spaces](https://arxiv.org/abs/2405.15413) | Shiyu Qin | arXiv 2024.05 | [![GitHub Repo stars](https://img.shields.io/github/stars/QinSY123/2024-MambaVC)](https://github.com/QinSY123/2024-MambaVC) |
|  | [Task-Aware Encoder Control for Deep Video Compression](https://openaccess.thecvf.com//content/CVPR2024/papers/Ge_Task-Aware_Encoder_Control_for_Deep_Video_Compression_CVPR_2024_paper.pdf) | Xingtong Ge | CVPR 2024 |  |

## Codec Performance
### Autoencoder based Coding Model
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
| VCT | [VCT: A Video Compression Transformer](https://proceedings.neurips.cc/paper_files/paper/2022/file/54dcf25318f9de5a7a01f0a4125c541e-Paper-Conference.pdf) | Fabian Mentzer | NeurIPS 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/google-research/google-research)](https://github.com/google-research/google-research/tree/master/vct) [![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/NeuralCompression)](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/torch_vct) |
| -- | [Conditional Entropy Coding for Efficient Video Compression](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620443.pdf) | Jerry Liu | ECCV 2020 | -- |
| -- | [Video Compression With Rate-Distortion Autoencoders](https://openaccess.thecvf.com/content_ICCV_2019/papers/Habibian_Video_Compression_With_Rate-Distortion_Autoencoders_ICCV_2019_paper.pdf) | Amirhossein Habibian | ICCV 2019 | -- |

### Hybrid Coding Model

#### Conditional Residual Coding
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [Conditional Residual Coding: A Remedy for Bottleneck Problems in Conditional Inter Frame Coding](https://ieeexplore.ieee.org/document/10416172) | Fabian Brand | TCSVT 2024 |  |

#### Conditional Coding
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [Prediction and Reference Quality Adaptation for Learned Video Compression](https://arxiv.org/pdf/2406.14118) | Xihua Sheng | arXiv 2024.06 |  |
| DCVC-FM | [Neural Video Compression with Feature Modulation](https://arxiv.org/abs/2402.17414) | Jiahao Li | CVPR 2024 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC-FM) |
|  | [Spatial Decomposition and Temporal Fusion based Inter Prediction for Learned Video Compression](https://ieeexplore.ieee.org/abstract/document/10416688) | Xihua Sheng | TCSVT 2024 |  |
|  | [Exploring Long- and Short-Range Temporal Information for Learned Video Compression](https://ieeexplore.ieee.org/document/10388053) | Huairui Wang | TIP 2024 |  |
|  | [Enhanced Context Mining and Filtering for Learned Video Compression](https://ieeexplore.ieee.org/document/10254316) | Haifeng Guo | TMM 2023 |  |
|  | [B-CANF: Adaptive B-frame Coding with Conditional Augmented Normalizing Flows](https://ieeexplore.ieee.org/abstract/document/10201921) | Mu-Jung Chen | TCSVT 2023 |  |
|  | [Neural Video Compression with Spatio-Temporal Cross-Covariance Transformers](https://dl.acm.org/doi/pdf/10.1145/3581783.3611960) | Zhenghao Chen | ACM MM 2023 |  |
| DCVC-DC | [Neural Video Compression with Diverse Contexts](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Neural_Video_Compression_With_Diverse_Contexts_CVPR_2023_paper.pdf) | Jiahao Li | CVPR 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC-DC) |
| DCVC-MIP | [Motion Information Propagation for Neural Video Compression](https://openaccess.thecvf.com/content/CVPR2023/papers/Qi_Motion_Information_Propagation_for_Neural_Video_Compression_CVPR_2023_paper.pdf) | Linfeng Qi | CVPR 2023 |  |
|  | [Hierarchical B-Frame Video Coding Using Two-Layer CANF Without Motion Coding](https://openaccess.thecvf.com/content/CVPR2023/papers/Alexandre_Hierarchical_B-Frame_Video_Coding_Using_Two-Layer_CANF_Without_Motion_Coding_CVPR_2023_paper.pdf) | David Alexandre | CVPR 2023 |  |
| MIMT | [MIMT: Masked Image Modeling Transformer for Video Compression](https://openreview.net/pdf?id=j9m-mVnndbm) | Jinxi Xiang | ICLR 2023 |  |
| DCVC-HEM | [Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression](https://dl.acm.org/doi/pdf/10.1145/3503161.3547845) | Jiahao Li | ACM MM 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC-HEM) |
| CANF-VC | [CANF-VC: Conditional Augmented Normalizing Flows for Video Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760193.pdf) | Yung-Han Ho | ECCV 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/NYCU-MAPL/CANF-VC)](https://github.com/NYCU-MAPL/CANF-VC) |
| DCVC-TCM | [Temporal Context Mining for Learned Video Compression](https://ieeexplore.ieee.org/abstract/document/9941493) | Xihua Sheng | TMM 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC-TCM) |
| DCVC | [Deep Contextual Video Compression](https://proceedings.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf) | Jiahao Li | NeurIPS 2021 | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DCVC)](https://github.com/microsoft/DCVC/tree/main/DCVC) |
| ELF-VC | [ELF-VC: Efficient Learned Flexible-Rate Video Coding](https://openaccess.thecvf.com/content/ICCV2021/papers/Rippel_ELF-VC_Efficient_Learned_Flexible-Rate_Video_Coding_ICCV_2021_paper.pdf) | Oren Rippel, Alexander G. Anderson | ICCV 2021 |  |

#### Residual Coding
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [Uncertainty-Aware Deep Video Compression With Ensembles](https://ieeexplore.ieee.org/abstract/document/10461131) | Wufei Ma | TMM 2024 |  |
|  | [Blurry Video Compression: A Trade-off between Visual Enhancement and Data Compression](https://openaccess.thecvf.com/content/WACV2024/papers/Argaw_Blurry_Video_Compression_A_Trade-Off_Between_Visual_Enhancement_and_Data_WACV_2024_paper.pdf) | Dawit Mureja Argaw | WACV 2024 |  |
| | [MPAI-EEV: Standardization Efforts of Artificial Intelligence based End-to-End Video Coding](https://ieeexplore.ieee.org/document/10234441) | Chuanmin Jia | TCSVT 2023 | [![Website](https://img.shields.io/badge/Visit-Website-green)](https://mpai.community/standards/mpai-eev/) [![GitHub Repo stars](https://img.shields.io/github/stars/yefeng00/EEV-0.4)](https://github.com/yefeng00/EEV-0.4) |
|  | [Learned Video Compression via Heterogeneous Deformable Compensation Network](https://ieeexplore.ieee.org/document/10163889) | Huairui Wang | TMM 2023 |  |
| ENVC | [Learning Cross-Scale Weighted Prediction for Efficient Neural Video Compression](https://ieeexplore.ieee.org/abstract/document/10159648) | Zongyu Guo | TIP 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/USTC-IMCL/ENVC)](https://github.com/USTC-IMCL/ENVC) |
|  | [Learned Video Compression With Efficient Temporal Context Learning](https://ieeexplore.ieee.org/document/10129217) | Dengchao Jin | TIP 2023 |  |
| | [Insights from Generative Modeling for Neural Video Compression](https://ieeexplore.ieee.org/abstract/document/10078276) | Ruihan Yang | TPAMI 2023 |  |
|  | [MMVC: Learned Multi-Mode Video Compression with Block-based Prediction Mode Selection and Density-Adaptive Entropy Coding](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_MMVC_Learned_Multi-Mode_Video_Compression_With_Block-Based_Prediction_Mode_Selection_CVPR_2023_paper.pdf) | Bowen Liu, Yu Chen | CVPR 2023 |  |
|  | [Boosting Neural Video Codecs by Exploiting Hierarchical Redundancy](https://openaccess.thecvf.com/content/WACV2023/papers/Pourreza_Boosting_Neural_Video_Codecs_by_Exploiting_Hierarchical_Redundancy_WACV_2023_paper.pdf) | Reza Pourreza | WACV 2023 |  |
|  | [DMVC: Decomposed Motion Modeling for Learned Video Compression](https://ieeexplore.ieee.org/document/10003249) | Kai Lin | TCSVT 2022 |  |
|  | [Advancing Learned Video Compression With In-Loop Frame Prediction](https://ieeexplore.ieee.org/abstract/document/9950550) | Ren Yang | TCSVT 2022 |  |
|  | [Edge-Based Video Compression Texture Synthesis Using Generative Adversarial Network](https://ieeexplore.ieee.org/document/9762281) | Chen Zhu | TCSVT 2022 |  |
|  | [End-to-End Neural Video Coding Using a Compound Spatiotemporal Representation](https://ieeexplore.ieee.org/document/9707786) | Haojie Liu | TCSVT 2022 |  |
|  | [AlphaVC: High-Performance and Efficient Learned Video Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790605.pdf) | Yibo Shi | ECCV 2022 |  |
| C2F | [Coarse-to-fine Deep Video Coding with Hyperprior-guided Mode Prediction](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Coarse-To-Fine_Deep_Video_Coding_With_Hyperprior-Guided_Mode_Prediction_CVPR_2022_paper.pdf) | Zhihao Hu | CVPR 2022 |  |
|  | [Learning-Based Video Coding with Joint Deep Compression and Enhancement](https://dl.acm.org/doi/pdf/10.1145/3503161.3548314) | Tiesong Zhao | ACMMM 2022 |  |
|  | [Structure-Preserving Motion Estimation for Learned Video Compression](https://dl.acm.org/doi/pdf/10.1145/3503161.3548156) | Han Gao | ACMMM 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/gaohan-12/SPME)](https://github.com/gaohan-12/SPME) |
|  | [End-to-End Rate-Distortion Optimized Learned Hierarchical Bi-Directional Video Compression](https://ieeexplore.ieee.org/abstract/document/9667275) | M. Akın Yılmaz | TIP 2021 |  |
| FVC | [FVC: A New Framework towards Deep Video Compression in Feature Space](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_FVC_A_New_Framework_Towards_Deep_Video_Compression_in_Feature_CVPR_2021_paper.pdf) | Zhihao Hu | CVPR 2021 | [![GitHub Repo stars](https://img.shields.io/github/stars/ZhihaoHu/PyTorchVideoCompression)](https://github.com/ZhihaoHu/PyTorchVideoCompression/tree/master/FVC) |
| RLVC | [Learning for Video Compression With Recurrent Auto-Encoder and Recurrent Probability Model](https://ieeexplore.ieee.org/abstract/document/9288876) | Ren Yang | JSTSP 2020 | [![GitHub Repo stars](https://img.shields.io/github/stars/RenYang-home/RLVC)](https://github.com/RenYang-home/RLVC) |
| DVC-Pro | [An End-to-End Learning Framework for Video Compression](https://ieeexplore.ieee.org/abstract/document/9072487) | Guo Lu | TPAMI 2020 |  |
| M-LVC | [M-LVC: Multiple Frames Prediction for Learned Video Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_M-LVC_Multiple_Frames_Prediction_for_Learned_Video_Compression_CVPR_2020_paper.pdf) | Jianping Lin | CVPR 2020 | [![GitHub Repo stars](https://img.shields.io/github/stars/JianpingLin/M-LVC_CVPR2020)](https://github.com/JianpingLin/M-LVC_CVPR2020) |
| SSF | [Scale-space flow for end-to-end optimized video compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.pdf) | Eirikur Agustsson | CVPR 2020 | [![GitHub Repo stars](https://img.shields.io/github/stars/InterDigitalInc/CompressAI)](https://github.com/InterDigitalInc/CompressAI) |
| RaFC | [Improving Deep Video Compression by Resolution-Adaptive Flow Coding](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470188.pdf) | Zhihao Hu | ECCV 2020 |  |
| DVC | [DVC: An End-to-end Deep Video Compression Framework](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.pdf) | Guo Lu | CVPR 2019 | [![GitHub Repo stars](https://img.shields.io/github/stars/GuoLusjtu/DVC)](https://github.com/GuoLusjtu/DVC) [![GitHub Repo stars](https://img.shields.io/github/stars/ZhihaoHu/PyTorchVideoCompression)](https://github.com/zhihaohu/pytorchvideocompression) [![GitHub Repo stars](https://img.shields.io/github/stars/RenYang-home/OpenDVC)](https://github.com/RenYang-home/OpenDVC) |
|  | [Learned Video Compression](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rippel_Learned_Video_Compression_ICCV_2019_paper.pdf) | Oren Rippel | ICCV 2019 |  |
| PMCNN | [Learning for Video Compression](https://ieeexplore.ieee.org/document/8610323) | Zhibo Chen | TCSVT 2019 |  |

### INR based Coding
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [Combining Frame and GOP Embeddings for Neural Video Representation](https://openaccess.thecvf.com/content/CVPR2024/papers/Saethre_Combining_Frame_and_GOP_Embeddings_for_Neural_Video_Representation_CVPR_2024_paper.pdf) | Jens Eirik Saethre | CVPR 2024 |  |
|  | [Boosting Neural Representations for Videos with a Conditional Decoder](https://arxiv.org/pdf/2402.18152.pdf) | Xinjie Zhang | CVPR 2024 |  |
| HiNeRV | [HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation](https://proceedings.neurips.cc/paper_files/paper/2023/file/e5dc475c370ff42f2f96dddf8191a40c-Paper-Conference.pdf) | Ho Man Kwan | NeurIPS 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/hmkx/HiNeRV)](https://github.com/hmkx/HiNeRV) |
|  | [Video Compression With Entropy-Constrained Neural Representations](https://openaccess.thecvf.com/content/CVPR2023/papers/Gomes_Video_Compression_With_Entropy-Constrained_Neural_Representations_CVPR_2023_paper.pdf) | Carlos Gomes | CVPR 2023 |  |
| HNeRV | [HNeRV: A Hybrid Neural Representation for Videos](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.pdf) | Hao Chen | CVPR 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/haochen-rye/HNeRV)](https://github.com/haochen-rye/HNeRV) |
| E-NeRV | [E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950263.pdf) | Zizhang Li | ECCV 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/kyleleey/E-NeRV)](https://github.com/kyleleey/E-NeRV) |
| NeRV | [NeRV: Neural Representations for Videos](https://proceedings.neurips.cc/paper_files/paper/2021/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf) | Hao Chen | NeurIPS 2021 | [![GitHub Repo stars](https://img.shields.io/github/stars/haochen-rye/NeRV)](https://github.com/haochen-rye/NeRV) |

### Framework-independent Optimization
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [Bit Allocation using Optimization](https://proceedings.mlr.press/v202/xu23c/xu23c.pdf) | Tongda Xu | ICML 2023 | [![GitHub Repo stars](https://img.shields.io/github/stars/tongdaxu/Bit-Allocation-Using-Optimization)](https://github.com/tongdaxu/Bit-Allocation-Using-Optimization) |

## Codec Scalability
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [LSSVC: A Learned Spatially Scalable Video Coding Scheme](https://ieeexplore.ieee.org/abstract/document/10521480) | Yifan Bian | TIP 2024 |  |
|  | [High Visual-Fidelity Learned Video Compression](https://dl.acm.org/doi/pdf/10.1145/3581783.3612530) | Meng Li | ACM MM 2023 |  |
|  | [DeepSVC: Deep Scalable Video Coding for Both Machine and Human Vision](https://dl.acm.org/doi/pdf/10.1145/3581783.3612500) | Hongbin Lin | ACM MM 2023 |  |
|  | [Neural Video Compression using GANs for Detail Synthesis and Propagation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860549.pdf) | Fabian Mentzer | ECCV 2022 |  |
|  | [Learning Based Multi-Modality Image and Video Compression](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Learning_Based_Multi-Modality_Image_and_Video_Compression_CVPR_2022_paper.pdf) | Guo Lu | CVPR 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/SZU-AdvTech-2022/165-Learning-based-Multi-modality-Image-and-Video-Compression)](https://github.com/SZU-AdvTech-2022/165-Learning-based-Multi-modality-Image-and-Video-Compression) |
|  | [LSVC: A Learning-Based Stereo Video Compression Framework](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_LSVC_A_Learning-Based_Stereo_Video_Compression_Framework_CVPR_2022_paper.pdf) | Zhenghao Chen | CVPR 2022 |  |
|  | [Video Coding using Learned Latent GAN Compression](https://dl.acm.org/doi/pdf/10.1145/3503161.3548219) | Mustafa Shukor | ACM MM 2022 |  |
|  | [Perceptual Learned Video Compression with Recurrent Conditional GAN](https://arxiv.org/pdf/2109.03082) | Ren Yang | IJCAI 2022 | [![GitHub Repo stars](https://img.shields.io/github/stars/RenYang-home/PLVC)](https://github.com/RenYang-home/PLVC) |

## Codec Practicality
| Models | Paper | First Author | Venue | Project |
| :--: | :---: | :--: | :--: | :--: |
|  | [MobileNVC: Real-Time 1080p Neural Video Compression on a Mobile Device](https://openaccess.thecvf.com/content/WACV2024/papers/van_Rozendaal_MobileNVC_Real-Time_1080p_Neural_Video_Compression_on_a_Mobile_Device_WACV_2024_paper.pdf) | Ties van Rozendaal | WACV 2024 |  |
|  | [Neural Rate Control for Learned Video Compression](https://openreview.net/pdf?id=42lcaojZug) | Yiwei Zhang | ICLR 2024 |  |
|  | [Sparse-to-Dense: High Efficiency Rate Control for End-to-end Scale-Adaptive Video Coding](https://ieeexplore.ieee.org/document/10246313) | Jiancong Chen | TCSVT 2023 |  |
|  | [Complexity-guided Slimmable Decoder for Efficient Deep Video Compression](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Complexity-Guided_Slimmable_Decoder_for_Efficient_Deep_Video_Compression_CVPR_2023_paper.pdf) | Zhihao Hu | CVPR 2023 |  |
|  | [FPX-NIC: An FPGA-Accelerated 4K Ultra-High-Definition Neural Video Coding System](https://ieeexplore.ieee.org/document/9745965) | Chuanmin Jia | TCSVT 2022 |  |

