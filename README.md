# README #

This README documents the necessary steps to run our tests.

### What is this repository for? ###

* A NEW denoising method that combines 3D Non-Local Means and LBP-TOP (local binary patterns on three orthogonal planes)
* Code version: 0.01-alpha
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### Dependencies ###

* Python 2.7
* OpenCV (for Python)
* NumPy
* SciPy
* scikit-image
* bob.ip.base (from [Bob](https://www.idiap.ch/software/bob/docs/releases/last/sphinx/html/index.html) library). See also this [link](https://github.com/idiap/bob/wiki/Dependencies)
* [Joblib](https://pythonhosted.org/joblib/parallel.html)

### TODO ###

* [DONE] - handle boundaries (use copyMakeBorder from OpenCV)
* [DONE] - dev the Non-Local Means and LBP-TOP denoising method
* [DONE] - create a script to run experiments
* [DONE] - implement a parallel version of Non-Local Means
* [DONE] - dev the Non-Local Means and MSB denoising method
* [DONE] - implement a 2D version of Non-Local Means
* implement a 3D version of Non-Local Means, which only considers 2D regions
* [DONE] - create a new test (a scene which lots of objects)
* implement a fast version of all NLM variants =)

### EXPERIMENTS ###

The file **runNonLocalMeans** run a filtering process with the user parameters:

* **-in** or **--input**: Input PATH with the input images (With Noise)
* **-out** or **--output**: Input PATH to save the output image (Denoised)
* **-ori** or **--originals**: Input PATH with the original images (Without Noise)
* **-H** or **--filterStrength**: Input the parameter H (Filter Strength)
* **-p** or **--patch**: Input the size of patch (Neighborhood size of centered voxel)
* **-w** or **--window**:Input the size of search window (Window size of centered voxel)
* **-msb** or **--MSBValue**: Value of Most Significant Bit
* **-seq** or **--sequence**: Sequence name of test (E.g. seq1)
* **-f** or **--folder**: Input PATH with database to save results (.db)

The file **runTests.sh** run the filtering process with more than one parameter: combining search window size **s = {7, 9, 11 and 13}** with patch window size **p = {7}**; search window size **s = {5}** with patch window size **p = {3}**; and all previous combinations with filter strength **H = {10, 15, 20 and 25}**. This file have four parameters:

* **-f** or **--folder**: base directory"
* **-s** or **--sigma**: sigma value for the gaussian noise"
* **-i** or **--image**: frame sequence folder (relative path)"
* **-m** or **--msb**: number of MSB to be considered"

Example how to run:

```
./runTests.sh -f /home/user/resultexp/ -i seq1 -s 10 -m 3

```

To execute the same examples that we run, execute the script **runAllTests.sh**. This script runs for each sequence dataset (with different levels of noise) the filtering process.

All the results will be save on table **results** on the database **3d_denoising_results.db**. The best values of filtering process can be get by the following code:


```
select seq, sigma, method, h, p, w, msb, psnr, max(ssim) 
from results 
where sigma == 25 and seq == 'seq4' and
method in ('NLM2D','NLM2D-LBP','NLM3D','NLM3D_viORI_texLBP_Adaptive','NLM3D_viORI_texLBP-MSB')
group by method 
order by max(ssim) desc;

```

The tests with BM3D-SAPCA [1] were made through the source code available in [Alessandro Foi personal webpage](http://www.cs.tut.fi/~foi/GCF-BM3D/BM3D.zip) in MATLAB using the same sequences of images. The sigma parameter used was the value of the noise level.

The algorithm can be run with the sample following code:

```
originalImage = imread('./dataset/seq1/original/gn-0-009625.png');
noiseImage = imread('./dataset/seq1/gaussian_noise-25/gn-25-009625.png');

[na denoisedImage] = BM3D(1, noiseImage, 25);

psnrV = PSNR(denoisedImage, originalImage);
ssimV = ssim(denoisedImage, originalImage);
```


### REFERENCES ###
[1] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, “Bm3d image denoising with shape-adaptive principal component analysis.” in SPARS’09-Signal Processing with Adaptive Sparse Structured Representations, Saint Malo, France, 2009.

