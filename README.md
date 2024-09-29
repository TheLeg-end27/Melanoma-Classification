# Melanoma-Classification
This is my second work to put into practice what was learned in the fast.ai course, Practical Deep Learning for Coders. The identify is to identify tumors, namely, Melanoma. This is a good starting point because of the extensive availability of images (dermoscopy) and clear differences between early to late stage tumors. More types of tumors to follow.

Run this command to install the libraries needed.

```
!pip install -Uqq fastai matplotlib
```
# Data collection
The images are taken from ISIC (International Skin Imaging Collaboration), which provides public datasets of dermoscopic images. The datasets used are:
https://api.isic-archive.com/collections/63/ 
https://api.isic-archive.com/collections/67/ 
https://api.isic-archive.com/collections/73/

The following image is a batch of images.

![sample](https://github.com/TheLeg-end27/Melanoma-Classification/blob/main/readme_images/sample.png)
# Model training
I've used google colab for the model training process, as this can require a lot of processing power. This process took longer due to the volume of images.

![finetune](https://github.com/TheLeg-end27/Melanoma-Classification/blob/main/readme_images/train.png)
![test](https://github.com/TheLeg-end27/Melanoma-Classification/blob/main/readme_images/test.png)
# Evaluation
With almost 90% accuracy, I expected the accuracy to be lower, as tumor detection can be quite difficult. This was a good place to start with tumor detection, but there will be more challenges as I explore more types of tumors.
