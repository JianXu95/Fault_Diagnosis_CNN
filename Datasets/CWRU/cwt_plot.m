load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\0\Normal.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\0\0.007-Ball.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\0\0.007-InnerRace.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\0\0.007-OuterRace.mat')

nt = 1024;

N =  X097_DE_time(1:nt);
BF07 = X118_DE_time(1:nt);
ORF07 = X130_DE_time(1:nt);
IRF07 = X105_DE_time(1:nt);

y = IRF07;

fs = 12000;
t = ([1:nt]-1)/fs;
wavename = 'cmor1-3'; %cmor3-3, db3
totalscal = 256;
wcf = centfrq(wavename);
cparam = 2*wcf*totalscal;
a = totalscal:-1:1;
scal = cparam./a;
coefs = cwt(y,scal,wavename);
f = scal2frq(scal, wavename, 1/fs);
imagesc(t,f,abs(coefs));
set(gca, 'YDir', 'normal')
colorbar;
% xlabel('时间 t/s');
% ylabel('频率 f/Hz');
% title('小波时频图');

% s1 = abs(coefs);
% s2 = imresize(s1,[32,32]);
% imagesc([1:32],[1:32],s2);
% colorbar;



