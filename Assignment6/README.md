# Assignment 6 submission: Detailed explanation
## Instructions

 - Same inputs, target outputs and initial weights which were discussed in the lecture were taken
 - The formula of every node and output of activation unit was written in terms of input, weights and output of nodes. Attaching screenshot for reference 
<img width="221" alt="Screenshot 2023-06-10 at 1 39 32 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/ef3fef84-cd06-4cbe-8344-2069c4860f51">

- The partial derivatives of the total error w.r.t every weight using chain rule were calculated. The calculations have been covered in the excel file. Attaching screenshot for reference
   -   <img width="628" alt="Screenshot 2023-06-10 at 1 41 06 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/74e9d7ec-9f64-490b-98a8-0f5134e00170">
- The first row was filled with formulas for all weights, node values, activated output of node values and the partial derivative of the total error w.r.t each weight
- Weights are were updated using updation formula for each weight Wnew = Wold - Learning_Rate*(dE/dWold) in the next row
- The formulas of the first row for the remaining rows and a graph of etotal was plotted
For the various learning rates, below are the attached screenshots
- 0.1

<img width="539" alt="Screenshot 2023-06-10 at 1 43 25 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/76fe97f9-0ecc-4957-adc7-6a0446b0ddca">


- 0.2

<img width="572" alt="Screenshot 2023-06-10 at 1 43 44 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/12b87a52-4764-4af6-ab4b-d97b26b907e4">


- 0.5

<img width="571" alt="Screenshot 2023-06-10 at 1 43 59 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/690ed404-cacf-4713-bd98-79f27c48e8b5">


- 0.8

<img width="555" alt="Screenshot 2023-06-10 at 1 44 09 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/98105cda-5103-45bc-b95a-cf46705dd5e4">

- 1

<img width="536" alt="Screenshot 2023-06-10 at 1 44 23 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/441c3192-9df9-40c3-9bff-025841661e97">

- 2

<img width="570" alt="Screenshot 2023-06-10 at 1 44 45 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/ea229ddf-2846-4f7f-8dc6-1ac85898e62f">


- From above images, it might be inferred that the more the learning rate, more faster is the rate with which the etotal reduces. But that is not the case. If the learning rate is exceedingly high then the Etotal might not converge to the minima. Following example of learning rate = 1000 proves this

<img width="563" alt="Screenshot 2023-06-10 at 1 49 57 AM" src="https://github.com/sagawritescode/ERA-V1/assets/45040561/de13a9db-f6f0-44c4-8c25-422330067f21">

