function [xs] = time2stft(x)

y = x;
fs = 12000;
win_sz = 64;
han_win = hanning(win_sz);      % Ñ¡Ôñº£Ã÷´°
nfft = win_sz;
noverlap = win_sz/2+2;
[S, F, T] = spectrogram(y, han_win, noverlap, nfft, fs);
Tx = T(1:32);
Fx = F(1:32);%Fx = flipud(Fx);
Sx = S(1:32,1:32);
Sx = flipud(Sx);
xs = reshape(Sx,[1 1024]);
