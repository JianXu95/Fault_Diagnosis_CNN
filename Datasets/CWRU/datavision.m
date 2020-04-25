load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\Normal.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.007-Ball.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.007-InnerRace.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.007-OuterRace.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.014-Ball.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.014-InnerRace.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.014-OuterRace.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.021-Ball.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.021-InnerRace.mat')
load('F:\Python\PycharmProjects\Fault_Diagnosis_CNN\Datasets\CWRU\3\0.021-OuterRace.mat')

nt = 12000;
n = 4096;

N =  X100_DE_time(1:nt);
BF07 = X121_DE_time(1:nt);
BF14 = X188_DE_time(1:nt);
BF21 = X225_DE_time(1:nt);
ORF07 = X133_DE_time(1:nt);
ORF14 = X200_DE_time(1:nt);
ORF21 = X237_DE_time(1:nt);
IRF07 = X108_DE_time(1:nt);
IRF14 = X172_DE_time(1:nt);
IRF21 = X212_DE_time(1:nt);
t = [0:11999]/120000;

% YN = fft(N,n);
% YBF07 = fft(BF07,n);
% YBF14 = fft(BF14,n);
% YBF21 = fft(BF21,n);
% YORF07 = fft(ORF07,n);
% YORF14 = fft(ORF14,n);
% YORF21 = fft(ORF21,n);
% YIRF07 = fft(IRF07,n);
% YIRF14 = fft(IRF14,n);
% YIRF21 = fft(IRF21,n);
% 
% PYN = 2*abs(YN)/n; PYN(1) = PYN(1)/2;
% PYBF07 = 2*abs(YBF07)/n; PYBF07(1) = PYBF07(1)/2;
% PYBF14 = 2*abs(YBF14)/n; PYBF14(1) = PYBF14(1)/2;
% PYBF21 = 2*abs(YBF21)/n; PYBF21(1) = PYBF21(1)/2;
% PYORF07 = 2*abs(YORF07)/n; PYORF07(1) = PYORF07(1)/2;
% PYORF14 = 2*abs(YORF14)/n; PYORF14(1) = PYORF14(1)/2;
% PYORF21 = 2*abs(YORF21)/n; PYORF21(1) = PYORF21(1)/2;
% PYIRF07 = 2*abs(YIRF07)/n; PYIRF07(1) = PYIRF07(1)/2;
% PYIRF14 = 2*abs(YIRF14)/n; PYIRF14(1) = PYIRF14(1)/2;
% PYIRF21 = 2*abs(YIRF21)/n; PYIRF21(1) = PYIRF21(1)/2;
% 
% f = ([1:n]-1)*12000/n;

% ---------------------- time domain -----------------
subplot(5,2,1);
plot(t,N);
subplot(5,2,2);
plot(t,BF07);
subplot(5,2,3);
plot(t,BF14);
subplot(5,2,4);
plot(t,BF21);
subplot(5,2,5);
plot(t,ORF07);
subplot(5,2,6);
plot(t,ORF14);
subplot(5,2,7);
plot(t,ORF21);
subplot(5,2,8);
plot(t,IRF07);
subplot(5,2,9);
plot(t,IRF14);
subplot(5,2,10);
plot(t,IRF21);
% -------------------------------------------








% --------------------   FFT  ----------------
% nf = 1000;
% subplot(10,1,1);
% plot(f(1:nf),PYN(1:nf));
% subplot(10,1,2);
% plot(f(1:nf),PYBF07(1:nf));
% subplot(10,1,3);
% plot(f(1:nf),PYBF14(1:nf));
% subplot(10,1,4);
% plot(f(1:nf),PYBF21(1:nf));
% subplot(10,1,5);
% plot(f(1:nf),PYORF07(1:nf));
% subplot(10,1,6);
% plot(f(1:nf),PYORF14(1:nf));
% subplot(10,1,7);
% plot(f(1:nf),PYORF21(1:nf));
% subplot(10,1,8);
% plot(f(1:nf),PYIRF07(1:nf));
% subplot(10,1,9);
% plot(f(1:nf),PYIRF14(1:nf));
% subplot(10,1,10);
% plot(f(1:nf),PYIRF21(1:nf));
% --------------------------------------------------------