close all; clear all; clc;
%importing the data file
T = readtable('spam.txt');

%extracting x(predictors) and y(target variable) from the table
x=T(:,1:57)
y=T(:,58:58)

%converting type of target variable to array
y=table2array(y)

%feature selection
[idx,scores] = fscmrmr(x,y);
%bar plot of the predictor importance scores.
figure
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')

%selecting top 20 predictors
C = templateTree;
tbl1 = T(:,T.Properties.VariableNames(idx(1:57)));
tbl2 = T(:,T.Properties.VariableNames(idx(1:20)));

%comparison for accuracies on both tables
[h,p] = testckfold(C,C,tbl1,tbl2,y)

%---kfold cv---%
kfold = 5;
fold=cvpartition(size(T,1),'kfold',kfold);

%Splitting and training data using kfold cv
for i=1:kfold

trainIdx=fold.training(i);
testIdx=fold.test(i);

x_train=tbl2(trainIdx,:);
y_train=y(trainIdx);

x_test=tbl2(testIdx,:);
y_test=y(testIdx);

random_forest = TreeBagger(20,x_train,y_train,'Method',"classification",'MinLeafSize',3,'MaxNumSplits',3,OOBPrediction="on");

%predicting on test data
pred = predict(random_forest,x_test);
[~,Scores1] = predict(random_forest,x_test);

%roc score
rocObj = rocmetrics(y_test,Scores1,random_forest.ClassNames);
rocObj.AUC
end

%error on training data
figure;
plot(oobError(random_forest))
title('Out-of-Bag Classification Error for Random Forest' );
xlabel("Number of Grown Trees")
ylabel("Out-of-Bag Classification Error")

%--Evaluation metrics--%

%roc curve
figure;
plot(rocObj,ClassNames=random_forest.ClassNames(1))
title('Roc curve for Random Forest; ROC score', rocObj.AUC(1) );

%converting type of pred from str to double
pred=str2double(pred)

%confusion matrix
figure;
confusionmatrix = confusionchart(y_test,pred);
title('Confusion Matrix for Random Forest' );
cm=confusionmat(y_test,pred);

%accuracy,precision,recall, F1-score
tp=cm(1);
fn=cm(2);
fp=cm(3);
tn=cm(4);
accuracy= (tp+tn)/(tp+tn+fp+fn);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
F1 = (2*precision*recall)/(precision+recall);
table(accuracy,precision,recall,F1,VariableNames=["Accuracy" "Precision" "Recall" "F1-score"])







