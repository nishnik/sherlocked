### convergence_check.py

convergence_check.py[] file runs the neural net for activation function combinations ```relu_relu_relu, relu_sigmoid_relu, 
relu_sigmoid_tanh ``` in the three layers with 50 epochs and plots the training and the test errors for each epoch.


### Result:

Both test and training data errors show that ```relu_relu_relu``` combination has higher convergence  than  ```relu_sigmoid_relu, 
 relu_sigmoid_tanh``` and  errors do not change much for  all the three combinations after 15 epoches.There is a sudden drop
in error rate for relu_sigmoid_relu from  12 to 13 epochs (Why?? Is this due to the sigmoid layer?? Need to be checked by
varying the learning rates of this combination.If drop in error rate occurs at lower  epoches for higher  learning rate, 
it can be confirmed that it is due to the sigmoid layer but if it is not the case then can't confirm anything.)


### comparing_sigmoid3vsrelu3.py

This file gives training and test errors for ``` relu_relu_relu, sigmoid_sigmoid_sigmoid ``` combinations for 10 epochs for
different values of learning rate. 

### Results:

Increasing the learning rate(lr), both the training and test errors of  ```  sigmoid_sigmoid_sigmoid ``` combination decrease. 
This is because, increasing the lr accelerates the convergence of the weights towards the local minimum by subtracting larger
values(either positive or negative ) from the weights during back propagation and thus also increasing the gradient during the 
next epoch.But if the learning rate is increased by a large value, the training and test errors start oscillating(slightly).(why??)

But this is not the case with ``` relu_relu_relu ```.Upon increasing lr, the training, test error increases.This can be a result 
of a large gradient flowing through a Relu node could cause the weights to update in such a way that the neuron  never activates
on any input again since the gradient would be zero from there after for . If this happens, then the gradient flowing through the
unit will forever be zero from that point on. That is, the Relu node can permanently become inactive.

These above results also indicate that performance of  ``` relu_relu_relu ``` network is better than other networks because  of 
it's linear, non-saturating form .So the gradient is never ```near zero``` and thus accelerates  the convergence of the weights.

