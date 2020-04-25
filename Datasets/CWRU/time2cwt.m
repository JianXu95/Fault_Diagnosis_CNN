function [xc] = time2cwt(x)

y = x;
wavename = 'cmor1-3'; %cmor3-3, db3
totalscal = 256;
wcf = centfrq(wavename);
cparam = 2*wcf*totalscal;
a = totalscal:-1:1;
scal = cparam./a;
coefs = cwt(y,scal,wavename);
s1 = abs(coefs);
s2 = imresize(s1,[32,32]);
xc = reshape(s2,[1 1024]);