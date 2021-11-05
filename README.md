# VGG_Sport_Clasiffication

Since 2014 VGG16 has been widely used for image classification. In this project i used VGG as the basic model with transfer learning. Transfer learning is a process where a model was already trained on a specific problem and can be used for solving other problems as well. The reason is that different tasks can be solved with similar features. This can be done by two approaches: The first approach is to freeze the pre-trained layers and train just the fully-connected layers, while the second approach is called fine-tuning when you can unfreeze some of the pre-trained layers and train them very gently with a very low learning rate. 

<img width="900" height="507" src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png" class="attachment-full size-full wp-post-image" alt="vgg16" loading="lazy">


My dataset contains various sports images from 22 categories, with 75 percent and 25 percent of the images allocated to training and validation, respectively. Aside from rescaling the images, data augmentation is also presented to prevent overfitting and improve model performance.
Randmoly has chosen the following images:


![image](https://user-images.githubusercontent.com/51881832/140488275-0af002e6-17ad-4e72-bffd-b390038cc227.png)


After 76 epochs the model showed accuracy of ~90%

![image](https://user-images.githubusercontent.com/51881832/140489055-b42a45ec-180b-4459-b16d-5e5eff05f024.png)

![image](https://user-images.githubusercontent.com/51881832/140489274-06e63c76-066e-4599-b25e-e58ae4bdec03.png)

![image](https://user-images.githubusercontent.com/51881832/140489356-010aeb37-f852-46d4-912b-fdf6717a2610.png)
