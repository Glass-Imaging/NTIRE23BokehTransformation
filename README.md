# [NTIRE 2023 Bokeh Effect Transformation Challenge](https://codalab.lisn.upsaclay.fr/competitions/10229)

Starter code for the NTIRE 2023 Bokeh Effect Transformation Challenge

> Many advancements of mobile cameras aim to reach the visual quality of professional full-frame cameras. Great progress was shown over the last years in optimizing the sharp regions of an image and in creating virtual portrait effects with artificially blurred backgrounds.
Now, the next goal of computational photography is to **optimize the bokeh effect** itself, which is the aesthetic quality of the blur in out-of-focus areas of an image. In professional full-frame photography, this effect is controlled by the aperture of the lens, the distance to the subject, and the focal length of the lens, but can also be adjusted in post-processing.
**The aim is to obtain a network design / solution capable of converting the the bokeh effect of one lens to the bokeh effect of another lens without harming the sharp foreground regions in the image.**
To the best of our knowledge, we are hosting the first challenge for this novel task and will be providing the first dataset and benchmark for it.

The top ranked participants will be awarded and invited to follow the CVPR submission guide for workshops to describe their solution and to submit to the associated NTIRE workshop at CVPR 2023. Challenge and Dataset report in CVPR proceedings. **More details in the [challenge website](https://codalab.lisn.upsaclay.fr/competitions/10229)**

<p align="center">
 <figure>
  <img src="media/bokeh-teaser.png" alt="Bokeh Teaser" width="800">
  <figcaption>Bokeh Transformation. Synthetic Data Example</figcaption>
</figure> 
</p>



## Contents
This repo contains
- a demo data set class for loading our training images
- a demo script for training a small model on these images
- the `metrics.py` file which shows how our 3 main metrics (PSNR, SSIM and LPIPS) are calculated

**We are open to PRs and Issues!**

## Dataset

The dataset consists of `id, source, target, alpha` tuples where the source and target share the same foreground and background, but the latter is artificially blurred with different lenses. For example: `00000, Sony50mmf1.8BS, Sony50mmf16.0BS, 35`. Additionally, we include the alpha mask of each image which can be useful for more advanced training losses or evaluations. Further, a `meta.txt` file lists the source and target lens and used disparity (as an indicator for blur strength) for each tuple. Check the [dataloader](https://github.com/Glass-Imaging/NTIRE23BokehTransformation/blob/main/examples/dataset.py).

## How to run
To run the demo, create a symbolic link like `ln -s DATA_PATH ./data` with DATA_PATH being the root dir of our unpacked zip file.
Then run `pip install -r requirements.txt` and `python3 -m examples.train`.

## Evaluation

Our evaluation is performed according to the following metrics:
- PSNR
- SSIM
- LPIPS (only during the final test phase)

<aside>
💡 The dataset in the test phase will include additional real-world captures using the same lenses as were simulated for the training data. These images will have the same structure of sharp foregrounds in front of blurred backgrounds.

</aside>

-----

## About the NTIRE Workshop at CVPR 2023

The 8th edition of NTIRE: New Trends in Image Restoration and Enhancement workshop will be held on June 18th, 2023 in conjunction with CVPR 2023.
The results of the challenge will be published at NTIRE 2023 workshop and in the CVPR 2023 Workshops proceedings.
More information about NTIRE workshop and challenge organizers is available here: https://cvlai.net/ntire/2023/

## Organizers

The Bokeh Effect Transformation Challenge is organized jointly with the NTIRE 2023 workshop in collaboration with [GLASS Imaging](https://glass-imaging.com/).

- **Marcos Conde (marcos.conde@uni-wuerzburg.de)***
- Radu Timofte (radu.timofte@uni-wuerzburg.de) *
- **Manuel Kolmet (manuel.k@glass-imaging.com)***
- Tom Bishop (tom@glass-imaging.com)

 
[*] are the contact persons of the NTIRE Bokeh Effect Transformation challenge.
