close all; clear all; clc;

%importing the data file
T = readtable('spam.txt');

% Partiion with 30% data as testing 
hpartition = cvpartition(size(T,1),'Holdout',0.3); 

% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = T(trainId,:);
testData = T(testId,:);

x_train=trainData(:,1:57);
y_train=trainData(:,58:58);
x_test=testData(:,1:57);
y_test=testData(:,58:58);

%-------- Model training ---------%

[lr,fit] = fitclinear(x_train,y_train,'Learner','logistic','Regularization','lasso','Solver','sparsa');
disp(lr)
disp(fit)

%predicting on test data
pred = predict(lr,x_test);

%converting data type of y-test and y_train to double below since type was table before
y_test = table2array(y_test);
y_train = table2array(y_train);

%classification score
[~,Scores] = predict(lr,x_test);
%size(Scores)

%roc curve
rocObj_lr = rocmetrics(y_test,Scores,lr.ClassNames);
%For a binary classification problem, the AUC values are equal to each other.
figure;
plot(rocObj_lr,ClassNames=lr.ClassNames(1));%plot for 1 class
title('Roc curve for Logistic regression; ROC score', rocObj_lr.AUC(1) );

%confusion matrix
figure;
Confusionmatrix = confusionchart(y_test,pred);
title('Confusion Matrix for Logistic regression' );

%accuracy,precision,recall, F1-score
cm=confusionmat(y_test,pred);
tp=cm(1);
fn=cm(2);
fp=cm(3);
tn=cm(4);
accuracy= (tp+tn)/(tp+tn+fp+fn);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
F1 = (2*precision*recall)/(precision+recall);
table(accuracy,precision,recall,F1,VariableNames=["Accuracy" "Precision" "Recall" "F1-score"])



