Libraries Required:
OpenCV
Keras
Tensorflow
matplotlib
Pandas

IDE used:
Kaggle Notebook (for building and training the model)
Jupyter Notebook (for implementing live web-cam)
Visual Studio Code (for building the user-interface)

Pre-requisties Required:
Python
Convolutional Neural Networks (CNN)
Multi-class Classifier Designing
Training a CNN model
Viola-Jones Algorithm
Flask

STEPS:
	For building the model, as a part of the first step all the necessary libraries were imported like the NumPy, Pandas, OpenCV, matplotlib, Keras, TensorFlow, seaborn etc.
	The dataset was read and the default shape and size of the images were observed. We assigned the labels to each emotion class. Some of the images from each class were plotted also.
	The next step is to apply some basic pre-processing techniques (like re-shaping to 224 x 224, converting to gray-scale etc.) on the images, i.e., to make the data compatible for neural networks. 
	The next task is to split the data into training and validation set and normalizing the results, as neural networks are very sensitive to unnormalized data. It was done by dividing the result by 255.0 and thus normalized results were obtained.
	We, then implemented the Transfer Learning concept which allows the model to be applied to a new problem using only pre-learned features, rather than reflecting changes in the training data to the model, while preserving the pre-trained weights. This is a technique often used in transfer learning applications. While developing a facial emotion recognition model using the VGG19 model, the pre-trained weights of the model are preserved, allowing it to be used in solving a new emotion recognition problem. In this way, it may be possible to obtain better results using less data. We visualized the different layers and the structure of the model. The input shape to the model was specified and we used Softmax as the activation function.
	We used two callbacks one is early stopping for avoiding overfitting training data and other ReduceLROnPlateau for learning rate. As the data in hand is less, we can easily use ImageDataGenerator. We used a batch size of 32 as it worked best for us and fixed the Epochs count to 25.
	The model was trained using the dataset and subsequently the accuracy and loss plot of the model and the confusion matrix was visualized using the matplotlib library.

This entire process from importing the libraries to training the model was done in a remote IDE named Kaggle Notebook as it had GPU installed. So, we were able to minimize the execution time through it. After the model is trained, we tested it using some images and subsequently the model.h5 file was downloaded from Kaggle. Our next set of tasks will be to use this model in various ways like in live web-cam implementation and in a web-application built using Flask.



Implementing Live webcam and predicting the emotion:
	The first step was to import the necessary libraries for our task like the OpenCV, NumPy, Keras, TensorFlow, matplotlib, Warnings.
	We loaded our model into the notebook that we already downloaded from the Kaggle notebook. The haarcascade classifier was also read. [We kept the model.h5 file and the haarcascade_frontalface_default.xml in our project folder.]
	The webcam feed was started. We can also pass any video path to it. The next idea is to capture the frames and returns a Boolean value and the captured image.
	The faces in the camera were detected. It was also successful in detecting multiple faces simultaneously. We took each face available on the camera and preprocess it, like drawing a rectangle over it.
	The next step is to crop the region of interest i.e., face area from the image and determine the emotion using the predict function of our model.
	We also displayed the predicted emotion on the screen to the user using OpenCV library. This entire task was basically an implementation of Viola-Jones algorithm.

Developing the web-application:
	The main idea behind developing the web-application was to present an interface to the users where they can upload any image and our model will be able to successfully detect the face and predict the true emotion conveyed.
	In addition to the previous libraries, a new library was also imported Flask. It is basically a web-application framework written in Python. 
	Two html pages were designed. The first one is index.html which is the landing page of the application where the user will upload the image. The second one is predict.html where the user will be redirected to after hitting the submit button in the previous page. 
	Similar, kind of approach to detect the faces from the images and cropping the region of interest was followed. 
	The model was then loaded and used to predict the dominant emotion conveyed in the chosen image. 
	Bootstrap was used to apply some styling techniques and make the application little presentable to the users.
