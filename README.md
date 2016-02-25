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
* bob.ip.base (from [Bob](https://www.idiap.ch/software/bob/docs/releases/last/sphinx/html/index.html) library)
* [Joblib](https://pythonhosted.org/joblib/parallel.html)

### TODO ###

* handle boundaries (use copyMakeBorder from OpenCV)
* dev the Non-Local Means and LBP-TOP denoising method
* implement a 2D version of Non-Local Means
* [DONE] - create a script to run experiments
* [DONE] - implement a parallel version of Non-Local Means
