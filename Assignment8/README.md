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

***Training Accuracy*** - 75.40%  </br>
***Testing Accuracy*** - 74.17%

I tried with different number of groups 4,8,16. As I increased the number of groups the accuracy kept improving slightly. The numbers and graphs I have submitted are of 16. The reason in increase of accuracy is the increase in the capacity and also scope for the model to adapt specifically to channels. More number of groups are better than 1 group (layer norm, results listed below) and it is comparable with the output of batchnorm, but training time can be more compared to batchnorm


#### Misclassfied Images:
<img width="463" alt="Screenshot 2023-06-24 at 3 41 41 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/3d5ee423-456d-43d4-a292-3190377025a1">

#### Misclassfied Images:
<img width="711" alt="Screenshot 2023-06-24 at 3 41 45 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/f59328cc-49bd-49f6-8ea9-86b1addbc5c7">




### Model 2 - Layer Normalisation Model 

Layer norm is the worst of all (comparatively) as it just classifies the channels in 1 group and normalises it. Higher number of groups marginally perform better than layer norm

***Training Accuracy*** - 73.49% </br>
***Testing Accuracy*** - 72.14%


#### Misclassfied Images:

<img width="461" alt="Screenshot 2023-06-24 at 3 22 17 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/28d8b757-fb23-4928-a717-739c6b38ba3d">

#### Misclassfied Images:
<img width="717" alt="Screenshot 2023-06-24 at 3 25 13 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/a8b6859e-3742-4f44-844a-7fe3ed708a07">





### Model 3 - Batch Normalisation Model 

Batch normalisation is the best of all in terms of accuracy


***Training Accuracy*** - 75% </br>
***Testing Accuracy*** - 74.74%


#### Misclassfied Images:

<img width="516" alt="Screenshot 2023-06-24 at 3 04 09 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/6328e4ed-1e1c-46b0-b06a-e08f10cedf8d">

#### Plots:

<img width="715" alt="Screenshot 2023-06-24 at 3 04 28 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/d1846c1f-840b-43d1-bc0c-88946e40aae0">


### Scope for experimentation

There was not much difference in accuracy due to which significant differences between all 3 approaches couldnt be pointed out. Reducing the number of parameters might have helped to figure out the nuances that might have been visible in the training of the model 
