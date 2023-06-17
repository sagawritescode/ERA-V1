# Assignment 7

## Objective

Attain 99.4 accuracy consistently under 15 epochs. Share the models used in the process from the beginning

## Models

### Model 1

**Target** - to get the code structure right 
**Result** - train - 99.87, test - 99.19
**Analysis** - There was no accuracy specific target with this model, but from the results it can be said that the model was clearly overfitting. 
The number of parameters been huge the model had an ability to overfit 

### Model 2 

**Target** - to commit to a skeleton. 
**Result** - train 99.29 test 99.86
**Analysis** - The model is giving a decent accuracy but the number of parameters are high. Need to reduce so that they are under the constraint of 8k

Intuition behind the model:
    After trying various architecture, I decided to commit to this architecture. 
    - Block 1 has 2 convolutional layers (instead of 1 that was in final model in class). Reason being I wanted Block 1 to have sufficient parameters to learn edges and gradients 
    - Transition block - helps reduce the numbers of parameters and implement squeeze 
    - Block 2 has 3 convolutional layers (instead of 4 in class final model). 3 layers seem sufficient enough to learn texture and patterns 
    -Global average pooling - I wanted this to be the part of the skeleton as I wanted to finalise the layers before proceeding. I tried experimenting with more/less and finalised the layers. Added GAP after finalising the layers i.e skeleton/architecture
   
### Model 3 
**Target** - to reduce the number of parameters under 8000
**Result** - train 98.05 test 97.78
**Analysis** - The accuracy is pretty less compared to desired 99+. Further experimentations required

Model summary

1 > 10 k = 3 p = 0 <br />
10 > 10 k = 3 p = 0 <br />
10 > 16 k = 3 p = 0 <br />
16 > 10 k = 1 p = 0 <br />
10 > 10  k = 3 p = 0 <br />
10 > 16  k = 3 p = 0 <br />
16 > 16 k = 3 p = 0 <br />
GAP (k = 5) <br />
16 > 10 k = 1  <br />


### Model 4
**Target** - To increase accuracy. Using batch normalization and dropout
**Result** - train 99.47 test 99.23
**Analysis** - The accuracy has improved significantly due to batch normalization. We applied a small value for dropout 1%. But overfitting still exists

## Model 5 
**Target** - To increase accuracy by using image augmentation. Increase dropout to reduce overfitting
**Result** - train 99.2 test 99.32 
**Analysis** - Used random crop and random rotation. Increased the dropout value to 0.03. The accuracy has improved but it doesnt cross 99.4. Had one instance where accuracy was touching 99.37 and touched 99.4. But it was less frequently replicable. Attaching screenshot of one such instance


![Screenshot 2023-06-17 at 3 37 36 AM](https://github.com/sagawritescode/ERA-V1/assets/45040561/335aa4bb-4229-4753-b6c4-0e0195eeb20e)

## Other attempts

Tried various other models. But none gave comparable accuracy to the above one 
- removed 10-10 layer in both convolution 1 and 2 block and tried to increase to 20 or 16. But it didnt work 
- tried with one layer less in convolution 1 block and one layer more in convolution 2 block 




