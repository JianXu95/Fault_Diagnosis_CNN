% ·ÂÕæÐÅºÅ
% parameters
f0 = 20;
f1 = 50;
f2 = 200;
f3 = 100;
f4 = 400;
f5 = 30;
f6 = 120;
a5 = 5;
a6 = 5;
% source signals
t = 0:0.0001:10;
s1 = (cos(2*pi*f0*t)+1).*sin(2*pi*f1*t);
s2 = (cos(2*pi*f0*t)+1).*sin(2*pi*f2*t);
s3 = sin(2*pi*f3*t);
s4 = sin(2*pi*f4*t);
s5 = (cos(2*pi*f0*t)+a5).*sin(2*pi*f5*t);
s6 = (cos(2*pi*f0*t)+a6).*sin(2*pi*f6*t);

% mixed signals
%A1 = rand(1,4);
A1 = [0.3 0.7 0.3 0.7];
X1 = A1*[s1;s2;s3;s4];
X1 = awgn(X1,30);
A2 = [0.3 0.7 0.3 0.7 0.5];
X2 = A2*[s1;s2;s3;s4;s5];
X2 = awgn(X2,30);
A3 = [0.3 0.7 0.3 0.7 0.5];
X3 = A3*[s1;s2;s3;s4;s6];
X3 = awgn(X3,30);
A4 = [0.3 0.7 0.3 0.7 0.4 0.4];
X4 = A4*[s1;s2;s3;s4;s5;s6];
X4 = awgn(X4,30);
plot(s6(1:10000));
%save('normal.mat','X1');
%save('fault1.mat','X2');
%save('fault2.mat','X3');
%save('fault12.mat','X4');
