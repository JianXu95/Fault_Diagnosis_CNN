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
win_sz = 64;
han_win = hanning(win_sz);      % 选择海明窗
nfft = win_sz;
noverlap = win_sz/2+2;
[S, F, T] = spectrogram(y, han_win, noverlap, nfft, fs);
Tx = T(1:32);
Fx = F(1:32);%Fx = flipud(Fx);
Sx = S(1:32,1:32);%Sx = flipud(Sx);
ss1 = reshape(Sx,[1 1024]);
ss2 = reshape(ss1,[32 32]);
imagesc(Tx, Fx, log10(abs(ss2)));
% imagesc(T, F, log10(abs(S)));
set(gca, 'YDir', 'normal')
colorbar;
% xlabel('时间 (secs)')
% ylabel('频率 (Hz)')
% title('短时傅里叶变换时频图')
