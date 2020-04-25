train = 100*load('train_acc.txt');
test =  100*load('test_acc.txt');
train1 =  100*load('train_acc1.txt');
test1 =  100*load('test_acc1.txt');

epoch = 1:1:80;

plot(epoch, train(1:80), epoch, test(1:80), epoch, train1(1:80), epoch, test1(1:80));