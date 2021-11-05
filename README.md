# VGG_Sport_Clasiffication

Since 2014 VGG16 has been widely used for image classification. In this project i used VGG as the basic model with transfer learning. Transfer learning is a process where a model was already trained on a specific problem and can be used for solving other problems as well. The reason is that different tasks can be solved with similar features. This can be done by two approaches: The first approach is to freeze the pre-trained layers and train just the fully-connected layers, while the second approach is called fine-tuning when you can unfreeze some of the pre-trained layers and train them very gently with a very low learning rate. 

<img width="900" height="507" src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png" class="attachment-full size-full wp-post-image" alt="vgg16" loading="lazy">


My dataset contains various sports images from 22 categories, with 75 percent and 25 percent of the images allocated to training and validation, respectively. Aside from rescaling the images, data augmentation is also presented to prevent overfitting and improve model performance.
Randmoly has chosen the following images:

<img width="342" alt="ללkא שם" src="https://user-images.githubusercontent.com/51881832/140489654-2523a19c-0d71-4ce6-a971-731b6935b6fb.png">


After 76 epochs the model showed accuracy of ~90%

<img width="554" alt="ללא שם6" src="https://user-images.githubusercontent.com/51881832/140489855-9c8be703-644a-48ce-93e7-e7a3582225d2.png">

<img width="202" alt="7ללא שם" src="https://user-images.githubusercontent.com/51881832/140489995-d20edac3-13f3-4339-ac04-b7b49c0a26cc.png">

<img width="200" alt="8ללא שם" src="https://user-images.githubusercontent.com/51881832/140490118-ca1543be-3107-4f7b-923e-00a38bf4f82d.png">

You can find my full project at: https://www.kaggle.com/omreekapon/sports-classification-using-vgg16-via-fine-tuning
