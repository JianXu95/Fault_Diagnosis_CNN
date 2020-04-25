% ·ÂÕæÐÅºÅ
% parameters
f0 = 20;
f1 = 50;
f2 = 200;
f3 = 100;
f4 = 400;
fri = 3000;
fro = 2000;
fi = 150;
fo = 80;
% source signals
t = 0:0.0001:20;
s1 = 0.1*(cos(2*pi*f0*t)+1).*sin(2*pi*f1*t);
s2 = 0.1*(cos(2*pi*f0*t)+1).*sin(2*pi*f2*t);
s3 = 0.5*sin(2*pi*f3*t);
s4 = 0.3*sin(2*pi*f4*t);
s5 = 2*sin(2*pi*fro*(mod(t,1/fo))).*exp(-300*pi*(mod(t,1/fo)));
s6 = 3*(cos(2*pi*f0*t)+ 1)/2.*sin(2*pi*fri*(mod(t,1/fi))).*exp(-500*pi*(mod(t,1/fi)));

% mixed signals
%A1 = rand(1,4);
A1 = [0.5 0.3];
X1 = A1*[s3;s4];
X1 = awgn(X1,20);
A2 = [0.5 0.3 1.0];
X2 = A2*[s3;s4;s5];
X2 = awgn(X2,20);
A3 = [0.5 0.3 1.0];
X3 = A3*[s3;s4;s6];
X3 = awgn(X3,20);
A4 = [0.5 0.3 1.0 1.0];
X4 = A4*[s3;s4;s5;s6];
X4 = awgn(X4,20);

subplot(2,1,1);
plot(t(1:2000), s5(1:2000));
xlabel('Time(s)');
ylabel('Amplitude');
subplot(2,1,2);
plot(t(1:2000), s6(1:2000));
xlabel('Time(s)');
ylabel('Amplitude');

% subplot(4,1,1);
% plot(t(1:2000), X1(1:2000));ylim([-4,4]);title('Normal');ylabel('Amplitude');
% subplot(4,1,2);
% plot(t(1:2000), X2(1:2000));ylim([-4,4]);title('Outer race fault');ylabel('Amplitude');
% subplot(4,1,3);
% plot(t(1:2000), X3(1:2000));ylim([-4,4]);title('Inner race fault');ylabel('Amplitude');
% subplot(4,1,4);
% plot(t(1:2000), X4(1:2000));ylim([-4,4]);title('Multiple fault');xlabel('Time(s)');ylabel('Amplitude');

% subplot(2,3,1);
% plot(t(1:2000), X1(1:2000));
% subplot(2,3,2);
% plot(t(1:2000), X2(1:2000));
% subplot(2,3,3);
% plot(t(1:2000), X3(1:2000));
% subplot(2,3,4);
% plot(t(1:2000), X4(1:2000));
% subplot(2,3,5);
% plot(t(1:2000), s3(1:2000));
% subplot(2,3,6);
% plot(t(1:2000), s4(1:2000));
save('normal.mat','X1');
save('outer.mat','X2');
save('inner.mat','X3');
save('multi.mat','X4');
