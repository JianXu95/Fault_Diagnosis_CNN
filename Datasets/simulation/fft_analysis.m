load('normal.mat');
x1 =  X1(1:2048);
y1 = fft(x1);
Py1 = 2*abs(y1)/2048;
subplot(2,2,1)
plot(Py1(1:200));
title('fft-normal');

load('outer.mat');
x2 =  X2(1:2048);
y2 = fft(x2);
Py2 = 2*abs(y2)/2048;
subplot(2,2,2)
plot(Py2(1:200));
title('fft-outer');

load('inner.mat');
x3 =  X3(1:2048);
y3 = fft(x3);
Py3 = 2*abs(y3)/2048;
subplot(2,2,3)
plot(Py3(1:200));
title('fft-inner');

load('multi.mat');
x4 =  X4(1:2048);
y4 = fft(x4);
Py4 = 2*abs(y4)/2048;
subplot(2,2,4)
plot(Py4(1:200));
title('fft-multi');