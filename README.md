# NTIRE 2023 Bokeh Effect Transformation Challenge
Starter code for the NTIRE 2023 Bokeh Effect Transformation Challenge

## Contents
This repo contains
- a demo data set class for loading our training images
- a demo script for training a small model on these images
- the `metrics.py` file which shows how our 3 main metrics (PSNR, SSIM and LPIPS) are calculated

## How to run
To run the demo, create a symbolic link like `ln -s DATA_PATH ./data` with DATA_PATH being the root dir of our unpacked zip file.
Then run `pip install -r requirements.txt` and `python3 -m examples.train`.
