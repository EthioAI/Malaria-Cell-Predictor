<p align="center">
  <img src="https://user-images.githubusercontent.com/69455299/181875921-018fe698-cea0-407f-b442-b4c7d9bbb8ec.jpg" width="1000" >
</p>

# Malaria Cell Predictor
This model predicts malaria from a cell. It is trained using the gradient boosting algorithm from xgboost library. The model is light weight and very fast relatively to other models I tried.<br/>

## How to setup?
1. You have to download the dataset using this link [here](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
2. Create a folder called data in the root directory.
3. Finally, extract the zip file into the data directory.


## How does it perform?
- Using ada boosting, the accuracy on train set was 86% but the deviation was kind of sparsed.
- Using gradient boosting, the accuracy on train set was 82% but the deviation was not sparsed.
- Gradient boosting was much faster than ada boosting and it also got about 80% score with the test set. So, I chose to use this cuz, the accuracy has no much different and gradient boosting is very fast and light weight.
- Ofcourse, this model was not to be trained using a classical machine learning algorithms. It require some deep learning algorithms which I will keep updating for the future.
