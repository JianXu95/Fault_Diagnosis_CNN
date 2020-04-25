% % input
% acc = [92.35 92.10 92.50 99.78 99.55 99.85];
% 
% way = {'SVM','kNN','ANN','VI+CNN','STFT+CNN','CWT+CNN'};
% 
% % show plot
% 
% bar(acc);
% set(gca,'XTickLabel',way)
% ylim([85,100]);
% xlabel('��Ϸ���'),ylabel('׼ȷ��(%)');

acc1 = [89.83 98.74 99.70 99.80 99.75];
time1 = [177.93 250.04 307.39 384.90 521.22];
ratio = {'10%','30%','50%','70%','90%'};
x = [1:5];
figure;
[AX,H1,H2] = plotyy(x,acc1,x,time1,'bar','plot');
set(AX(1),'yLim',[80,100],'fontsize',14);  %�������Y��Ŀ̶�
set(AX(2),'yLim',[0,600],'fontsize',14) %�����ұ�Y��Ŀ̶�
set(get(AX(1),'Ylabel'),'string','׼ȷ��(%)','fontsize',14);
set(get(AX(2),'Ylabel'),'string','ѵ��ʱ��(s)','fontsize',14);
% set(AX(1),'xTick',[]);
set(AX(2),'xTick',[]);
set(gca,'XTickLabel',ratio);
xlabel('ѵ����������');

