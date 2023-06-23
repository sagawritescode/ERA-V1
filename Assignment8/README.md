# Assignment 8

## Code Explanation:

We have a model.py file which contains a base model called `NormalisationModel` which abstracts the normalisation layer for each CNN layer. This abstraction is in the form of a function called `getNormalisationLayer` which could easily belong to a class. The 3 models are derived from this batch class as the architecture is common for them.
There are no other files as the functions are directly used in the colab notebook and can be found there. The code can definitely be improved and abstracted into different files (scope for improvement) </br>

In the colab file, we follow a similar structure as previous classes. Download and transform data, define util functions and train the model. The initial part of the code (reading data and displaying) was inspired by the pytorch tutorial on cifar10 ([link](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html]))

After training we display misclassified images and plot the accuracies and losses

## Comparing the models and learnings

(Note: Left label is predicted by model and right label is the correct label) </br>

All models were trained on 15 epochs. The number of parameters was 39,056.  
Below are the findings:

### Model 1 - Group Normalisation Model 


### Model 2 - Layer Normalisation Model 

***Training Accuracy*** - 73.49% </br>
***Testing Accuracy*** - 72.14%


#### Misclassfied Images:

<img width="461" alt="Screenshot 2023-06-24 at 3 22 17 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/28d8b757-fb23-4928-a717-739c6b38ba3d">

### Plots:
<img width="461" alt="Screenshot 2023-06-24 at 3 22 17 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/be8afce2-7f51-418d-b61f-e711f95143a2">




### Model 3 - Batch Normalisation Model 


***Training Accuracy*** - 75% </br>
***Testing Accuracy*** - 74.74%


#### Misclassfied Images:

<img width="516" alt="Screenshot 2023-06-24 at 3 04 09 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/6328e4ed-1e1c-46b0-b06a-e08f10cedf8d">

#### Plots:

<img width="715" alt="Screenshot 2023-06-24 at 3 04 28 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/d1846c1f-840b-43d1-bc0c-88946e40aae0">


