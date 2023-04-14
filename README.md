# Background
This project is a pytorch project of the algorithm FCD-GAN. Fully Convolutional Change Detection Framework with Generative Adversarial Network (FCD-GAN) is a newly proposed framework for change detection in multi-temporal remote sensing images. The corresponding publication can be found in the following citation:

C. Wu, B. Du, and L. Zhang, “Fully Convolutional Change Detection Framework with Generative Adversarial Network for Unsupervised, Weakly Supervised and Regional Supervised Change Detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1-15, 2023.

In this project, three demo codes are released for unsupervised, weakly supervised and regional supervised change detection, as "Demo_USSS", "Demo_WSSS", and "Demo_RSSS".

# Install
This project uses 'gdal', 'numpy', 'tqdm', 'cv2', 'pil', 'skimage'. Go check them out if you don't have them locally installed.

# Usage
The implementations of the algorithms are introduced in this section.

## Demo_USSS
This code is a demo for the algorithm of unsupervised change detection.
The user should input the bi-temporal images with the following code:
```python
dir = r'/data'
ImageXName = 'T1.tif'
ImageYName = 'T2.tif'
RefName = 'ref.tif'·
```
In order to customize the reference, the user can define the value to indicate change or nonchange in the reference image with the following code:
```python
gt_map = [1, 2]
```
This code will output two images (the color image is optional) and one txt file. 
The first image is a change density image, with the range of [0, 1]. Higher value indicates higher probability to be changed. This image will be outputed in the following path:
```python
outdir = dir
ext = '_l1w065_pw04_github'
CMapName = 'ChangeDensity{}'.format(ext)
```
The second image is a colorful evaluation image, where the value of {0, 1, 2, 3} indicates {FP, FN, TP, TN}. The density image is firstly assigned to be unchanged (0) or changed (1) with the given prob_threshold in the code, and evaluated with the reference to produce the colorful evaluation image is optional with the following switch:
```python
write_color = True
```
The colorful evaluation image will be outputed in the following path:
```python
OutColorPath = os.path.join(outdir, "{}_acc_color{}".format(CMapName, ext1))
```
The last output file is a txt file, which will record the parameter settings of this code. The txt file will be outputed in the following path:
```python
ParaTxtPath = os.path.join(outdir,'Para_{}{}.txt'.format(time.strftime("%b%d%H%M", time.localtime()), ext))
```
