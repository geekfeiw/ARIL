# APL

Code and Data of the paper, [Joint Activity Recognition and Indoor Localization with WiFi Fingerprints]().
 
**APL** is for **A**ctivity recognition **P**lus indoor **L**ocalization.

 

# Evaluated Environment 
1. PyTorch 1.0.0
2. Python 3.7
3. CUDA 8


# Hardware: Ettus N210 and Ettus Clock

![hardware](figs/hardware.png)


# Usage

1. Please download [data](https://drive.google.com/open?id=1SCxUHbl6rNWM3kT0c-D4s_kyAero9_-o
), and decompress it at the root folder of this repository.

> Activity Label: 0. hand up;  1. hand down; 2. hand left; 3. hand right; 4. hand circle; 5. hand cross.
> Location Label: 0, 1, 2, ..., 15


2. Please download [pre-trained weights](https://drive.google.com/open?id=1UT61Gs746yijxiKvLyP0wHIPxi9MYz0Y), and decompress it at the root folder of this repository.

3. Then run train.py or test.py


# For [t-SNE](https://lvdmaaten.github.io/tsne/) visualization

Please download [vis](https://drive.google.com/open?id=1Vd3Bs8V95XzwsJd5Y4cRVN6bimYt5Do4
), and run **main_plot_tsne.m**

1. t-SNE visualization for activity recognition 
![tsne_act](figs/apl_tsne_act.png)

2. t-SNE visualization for indoor localization 
![tsne_loc](figs/apl_tsne_loc.png)



If this helps your research, please cite this paper.

```

```
