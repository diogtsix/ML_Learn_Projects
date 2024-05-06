# **ML_Learn_Projects**

This repository contains multiple different projects regarding **Machine Learning** and **Deep Learning**

Currently this repositor Consists : 

- ML_Learn_Project : An introductory repository on how to use specific ML libraries and algorithms. This repository serves educational and learning purposes. 

Included in this project :

                            1) ML_Supervised_GammaTelescope
                            2) ML_Supervised_SeoulBikeSharing
                            3) ML_Unsupervised_Seeds

# **DeepLearning_Learn_Project**

- DeepLearning_Learn_Project :Multiple Different Projects bulding Neural Network and Machine learning models. Projects utilizing both libraries (TensorFlow, Keras, Pandas.. etc.) and Analytical deriviations of machine learning algorithms.

Included in this project : 

                            1) NeuralNetwork_HandWritten_Digits_Introduction
                            2) Gradient_Descent_Analytical (Implementation without libraries)
                            3) Neural_Network_Analytical
                            4) Stochastic_vs_Batch_GD_Analytical
                            5) ANN_Prediction_Customer_churn
                            6) Dropout_Regularization_Neural_Network
                            7) Handling Imballanced Data (5 Methods to handle Imballanced Data)
                            8) Convolutional_Neural_Network (For Image dataset)
                            9) CNN_Data_Augmentation (Data augmentation to avoid overfitting)
                            10) RNN_Word_Embeddings (RNN model for word to vector modelling, training and predictions)
                            11) TensorFLow_Input_Pipeline (Tensorflow Pipeline, cache, prefetch etc.)
                            12) Quantization_in_Deep_Learning (Quantization for use in EDGE devices, size reduction)


# **Potato_Disease_App**

- Trained a CNN model for image classification. We classify potato leafs based on type of disease (Early_blight, Healthy, Late_blight). Used FastApi for local host generation and interaction with the model via web. 

- In order to use this model run and train the model from CNN_model_build.ipynb and then run main.py and find your localhost onlne to upload a desired image. For capacity purposes images are not included in this repository and you need to download them from the provided link inside the CNN_model_build.ipynb. 

- Before running the project install the dependences from requirements.

- File-Folder description : 

                            1) api folder : contains the py files to run and create the localhost server
                            2) model_build : contains the notebook that trains the model
                            
# **Libraries Needed**

The following libs are used on all 3 projects. 

- tensorflow
- pandas
- Numpy
- sklearn
- seaborn
- matplotlib
- fastapi
- uvicorn
- python-multipart
- pillow
- tensorflow-serving-api
