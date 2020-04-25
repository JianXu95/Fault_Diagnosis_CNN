acc=[98.75,96.88, 95.63, 98.75, 98.13, 100];
bar(acc);
ylabel('Test Accuracy (%)')
xlim([0.5,6.5]);
ylim([94,100]);
set(gca,'XTicklabel',{'SVM','BPNN','2-D CNN','1-D CNN','WDCNN','LDCNN'});
