close all; clear all; clc;

%importing the data file
T = readtable('spam.txt');

%extracting the target column from table
y=T.label;
rng default

%---kfold cv---%
kfold = 5;
fold=cvpartition(size(T,1),'kfold',kfold);

%Splitting and training data using kfold cv
for i=1:kfold

trainIdx=fold.training(i);
testIdx=fold.test(i);

x_train=T(trainIdx,:);
y_train=y(trainIdx);

x_test=T(testIdx,:);
y_test=y(testIdx);

[lr,fit] = fitclinear(x_train,y_train,'Learner','logistic','Regularization','ridge','Solver',{'sgd' 'lbfgs'});
disp(lr)
disp(fit)

%predicting on test data
pred = predict(lr,x_test);
[~,Scores1] = predict(lr,x_test); 
end

%------Evaluation metrics------

%roc-curve
rocObj_lr = rocmetrics(y_test,Scores1,lr.ClassNames);
figure;
plot(rocObj_lr,ClassNames=lr.ClassNames(1));
title('Roc curve for Logistic regression; ROC score', rocObj_lr.AUC(1) );

%for training
[~,Scores2] = predict(lr,x_train)
rocObj_lr = rocmetrics(y_train,Scores2,lr.ClassNames);
figure;
plot(rocObj_lr,ClassNames=lr.ClassNames(1));
title('Roc curve for training Logistic regression; ROC score', rocObj_lr.AUC(1) );


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
