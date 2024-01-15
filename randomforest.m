close all; clear all;
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

%-----------Model Training----------%

%Setting the random number generator to default for reproducibility.
rng("default")

random_forest = TreeBagger(20,x_train,y_train,Method="classification",OOBPrediction="on")

%predicting on test data
pred = predict(random_forest,x_test);
[~,Scores1] = predict(random_forest,x_test);

%checking the Predicted and actual labels
table(y_test,pred,VariableNames=["TrueLabel" "PredictedLabel"])
y_test = table2array(y_test);
y_train = table2array(y_train)


%--Evaluation metrics--%
%error on training data
figure;
plot(oobError(random_forest))
title('Out-of-Bag Classification Error for Random Forest' );
xlabel("Number of Grown Trees")
ylabel("Out-of-Bag Classification Error")

%roc curve
rocObj = rocmetrics(y_test,Scores1,random_forest.ClassNames);
figure;
plot(rocObj,ClassNames=random_forest.ClassNames(1))%plot for 1 class
title('Roc curve for Random Forest; ROC score', rocObj.AUC(1) );

%converting pred type from str to double
pred=str2double(pred);

%confusion matrix
figure;
confusionmatrix = confusionchart(y_test,pred);
title('Confusion Matrix for Random Forest' );

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


