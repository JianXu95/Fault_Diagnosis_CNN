Data = load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\data\7\-4db\dataset1024.mat');
train = Data.X_train;
test = Data.X_test;
[r1, c1] = size(train);
X_train = zeros(r1,c1);
y_train = Data.y_train;
for i = 1:r1
    X_train(i,:) = time2cwt(train(i,:));
end

[r2, c2] = size(test);
X_test = zeros(r2,c2);
y_test = Data.y_test;
for j = 1:r2
    X_test(j,:) = time2cwt(test(j,:));
end

% save('cwt\dataset1024_c.mat','X_train','y_train','X_test','y_test');
save('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\data\7\-4db\dataset1024_c.mat','X_train','y_train','X_test','y_test');