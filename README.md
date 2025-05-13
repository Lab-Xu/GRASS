## GRASS



Spatial Transcriptomics (ST) technology enables the simultaneous capture of gene expression profile and spatial information within two-dimensional tissue slices. However, conventional analyses that process each individual slice independently often overlook shared features across multiple slices, limiting comprehensive biological insights. To address this, we introduce GRASS, a deep graph representation learning-based framework designed for the integration and alignment of multi-slice ST data. GRASS consists of two core modules: GRASS\_Integration, which employs a heterogeneous graph architecture integrating contrastive learning and a multi-expert collaboration strategy to fully utilize both shared and unique information, enabling multi-slice integration, clustering, and various downstream analyses; and GRASS\_Alignment, which uses a dual-perception similarity metric to guide spot-level alignment, supporting downstream tasks such as imputation and three-dimensional (3D) reconstruction. Experimental results on six ST datasets from five different platforms demonstrate that GRASS consistently outperforms six state-of-the-art (SOTA) methods in both integration and alignment tasks. By comprehensively addressing multi-level information integration, GRASS emerges as an ideal solution for the joint analysis of multi-slice ST data.



## Environment installation



**Note**: The current version of GRASS supports Linux and Windows platform.

Install packages listed on a pip file:

```
pip install -r requirements.txt
```

Install `rpy2` package:

```
pip install rpy2==3.5.10
```

Please note that the R language and the mclust package need to be installed on your system.

Install the corresponding versions of pytorch and torch_geometrics:

```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_geometric==2.0.4
```



## Run the code

All code is currently launched through `GRASS_Intergration_Alignment_DLPFC.ipynb`.

