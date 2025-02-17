## GRASS







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

All code is currently launched through `python main.py`.

