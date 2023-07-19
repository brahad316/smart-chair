# Smart Chair

## ML classifier and analysis

We used different ML algorithms to train a classifier model to classify a test subject's posture. Here is how different models performed.

### Tree

Max number of splits: 100
Split criterion: Gini's diversity index
Surrogate decision splits: off

Confusion matrix:

![image](https://github.com/brahad316/smart-chair/assets/94699627/0bb88a64-8791-4722-a392-bd9a582d8bc2)


Results:

![image](https://github.com/brahad316/smart-chair/assets/94699627/db2f976d-7f72-4957-ba6e-5ef6f1d9c012)


### Support Vector Machine (SVM)

* Linear SVM

Confusion matrix:

![image](https://github.com/brahad316/smart-chair/assets/94699627/4ecc161b-46b0-46b5-8798-48da86443c16)


Results:

![image](https://github.com/brahad316/smart-chair/assets/94699627/e96606af-f1fe-4729-96bc-6ed0e1a591e7)

* Gaussian SVM

Confusion matrix:

![image](https://github.com/brahad316/smart-chair/assets/94699627/90b55015-7017-42d8-8bf0-2c68ec29d5e9)


Results:

![image](https://github.com/brahad316/smart-chair/assets/94699627/26fe74e6-9909-43ff-ad22-a54335f4326f)

### K Nearest Neighbours (KNN)

Number of nearest neighbours: 3
Distance metric: Euclidean

Confusion matrix:

![image](https://github.com/brahad316/smart-chair/assets/94699627/4726a2a3-81f1-4a03-9677-bd277ae903fe)

Results:

![image](https://github.com/brahad316/smart-chair/assets/94699627/5f2d4fd7-f453-461f-99ad-f4e78c88823f)

### Neural Networks

* network 1:

number of layers: 3
first layer size: 100
second layer size: 25
third layer size: 10
activation: ReLU

Confusion matrix:

![image](https://github.com/brahad316/smart-chair/assets/94699627/c6efd2a8-6468-4f18-adc8-fa686f7033c0)

Results:

![image](https://github.com/brahad316/smart-chair/assets/94699627/faee3907-a96e-4209-8dd7-e20334433e3f)

* network 1:

number of layers: 3
first layer size:
second layer size:
third layer size:
activation: ReLU

Confusion matrix:

![image](https://github.com/brahad316/smart-chair/assets/94699627/b21f2b43-65bf-4291-ae66-e2df861120b3)

Results:

![image](https://github.com/brahad316/smart-chair/assets/94699627/150b112a-a17c-4348-b0c9-5f8b22f379ba)


` Note that all the confusion matrices are from validation.

> For our use case it's best to have the model with least training time. Thus we 
