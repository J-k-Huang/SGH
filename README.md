# SGH
## Environment
* python   3.5.4  
* pytorch  1.0.0  
* numpy    1.15.2  
* torchvision  0.6.1  
## Benchmark
* Digits contains three datasets: MNIST [here](https://github.com/thuml/CDAN/tree/master/data), USPS [here](https://github.com/thuml/CDAN/tree/master/data), and SVHN [here](https://drive.google.com/file/d/1Y0wT_ElbDcnFxtu25MB74npURwwijEdT/view). 
* Office-31 contains 4652 images across 31 classes from three domains: Amazon (A), DSLR (D), and Webcam (W). Office-31 dataset can be found [here](https://faculty.cc.gatech.edu/~judy/domainadapt/)
* Office-Home contains 15 500 images of 65 classes from four domains: Ar, Cl, Pr, and Rw. Office-Home dataset can be found [here](https://www.hemanthdv.org/officeHomeDataset.html)
* VisDA-2017 is a simulation-to-real dataset for domain adaptation over 280 000 images across 12 categories. VisDA-2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public)
## Demo  
Train on CDAN on Digits dataset
```  
cd CDAN+SGH/demo_digital
python train_mnistusps_CDAN.py CDAN
```
Train on CDAN on image dataset (Office-31/Office-home/VisDA-2017)
```  
cd CDAN+SGH/demo_images
python train_image_AW.py CDAN
```
Train on DWL on Digits dataset
```  
cd DWL+SGH/demo_digital
python main_grad.py
```
Train on DWL on image dataset
```  
cd DWL+SGH/demo_office31
python main_DW.py
```
## Contact  
If you have any problem about our code, feel free to contact jkhuang@cqu.edu.cn
