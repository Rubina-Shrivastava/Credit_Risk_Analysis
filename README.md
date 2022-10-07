# Credit_Risk_Analysis
## Purpose:
### Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans.In this project we want to take a look at how all the factors in our loan_stats csv help predict whether someone is low or high risk status. One method that data scientists use for this type of issue is creating a model and then evaluate and train the models that they create. In this specific project we are using imbalanced-learn and scikit-learn libraries to build models and evalute them using a resampling method. We focus on Supervised Learning using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk. This reason why this is called "Supervised Learning" is because the data includes a labeled outcome , algorithms include RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN, BalancedRandomForestClassifier, and EasyEnsembleClassifier.
## Results:
### Using our knowledge of the imbalanced-learn and scikit-learn libraries, I evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First,  I use the oversampling RandomOverSampler and SMOTE algorithms, and then I use the undersampling ClusterCentroids algorithm. Using these algorithms, I resampled the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
### An accuracy score for the model is calculated
### A confusion matrix has been generated
### An imbalanced classification report has been generated 
#### Naive Random Oversampling results: Our balanced accuracy test it 67%, the precision for the high_risk has a very low positivity at 1% and the recall is 74%
Image1
### Oversampling
#### RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.
Image2
### Undersampling
#### lusterCentroids Model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk
Image3
###  In deliverable 3 we are  Useing Ensemble Classifiers to Predict Credit Risk to compare two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.for this deliverable in place of jupyter Notebook I used Google Colab.
Image4
### The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%."Low Risk" still had a precision rate of 100% with the recall now at 94%.
Image5
## Summary:
### In the first four models we undersampled, oversampled and did a combination of both to try and determine which model is best at predicting which loans are the highest risk. The next two models we resampled the data using ensemble classifiers to try and predict which which loans are high or low risk. In our first four models our accuracy score is not as high as the ensemble classifiers and the recall in the oversampling/undersampling/mixed models is low as well. Typically in your models you want a good balance of recall and precision which is why I recommend the ensemble classifiers over the first four models.In reviewing all six models, the EasyEnsembleClassifer model yielded the best results with an accuracy rate of 93.2% and a 9% precision rate when predicting "High Risk candidates. The sensitivity rate (aka recall) was also the highest at 92% compared to the other models.The recall score also needs to fall within 0 and 1, with numbers closer to 1 being the better model. The Easy Ensemble AdaBoost Classifier had the highest recall score, making it the final best machine learning model to choose for further credit card analysis.