
# ERA Assignment 5 

## Problem Statement 

- Restructure the solution of Assignment 4 into following files:
  - model.py
  - utils.py
  - S5.ipynb 

- Add a readme file explaining details about the code 


## Code Explanation 

### Overview

This is a repo containing code for classification of MNIST dataset using neural networks at an introductory level. It covers basics of pytorch, dealing with datasets, how to define a model and train it and finally how to present the results. Read further for detailed breakdown of the code  

#### model.py 

This file contains the model class **Net** which consists of the model definition i.e architecture of our model. 
The reason for segragating the class in a different file is to have clear and simple separation of code. 
One can easily look up the model details by opening this file rather than scrolling in a long ipynb file.
That is also one of the reasons why this file only contains the class definition of the model and not anything else as the name suggests 

#### utils.py 

This file contains functions that are generic enough. In general, it is preferred to keep stateless or simple transformation functions here and other class specific functions in class files. Note that its not true in each case 

##### How it is different from the original colab file? 
- I have moved the `GetCorrectPredCount`, `train`, `test` to the utils file as they are equivalent to utilities and are reusable. I had to pass extra arguments to train and test so that it transforms the accuracy and loss list appropriately i.e append each computation to it. 
- I moved the plotting logic into a separate function `plot_loss_and_accuracy` which can be used in S5.ipynb. I had to pass losses and accuracies to this function as arguments 

#### S5.ipynb

This is the main file/notebook to be run. It has been refactored to reflect imports from model and utils. Run this file to train the model and get the results
