% This script runs the simulations presented in Higgins et al 2022

%% Preliminary:

% setup paths and figure directories:
figdir = '/Users/chiggins/Documents/Figures_RepDynamics/';
mkdir(figdir)
addpath(genpath('/Users/chiggins/Documents/MATLAB/RepresentationalDynamicsModelling'));

%% Figure 1:

% Here we simulate the displacement and velocity of a pendulum

% For the left hand side component of this figure, 
t=0.01:0.01:1;

x = sin(2*pi*t);
x_dash = cos(2*pi*t);

figure('Position',[440 462 285 336]);
subplot(2,1,1);
plot(t,x,'k','LineWidth',2);
plot4paper('Time','Displacement');
hold on;
plot(t,x_dash,'r','LineWidth',2,'LineStyle','--');

% and plot information content associated with this:
sigma = 0.5;
for i=1:length(t)
    alphaterm(i,1) = 2*x(i).^2/sigma;
    alphaterm(i,2) = 2*[x(i),x_dash(i)]*inv(diag([sigma,sigma]))*[x(i),x_dash(i)]';
end
subplot(2,1,2);
plot(t,alphaterm(:,1),'k','LineWidth',2);
plot4paper('Time','Info Content');
set(gca,'YTick','')
ylim([0,alphaterm(1,2)*1.2]);
hold on;
plot(t,alphaterm(:,2),'r','LineWidth',2,'LineStyle','--');
print([figdir,'Fig1'],'-dpng');


%% Figure 2: 

% Simulate the examples shown in figure 2:

% Example 1:
t = 0.001:0.001:0.1;
s = 0.5; % noise magnitude
f = 10; % frequency of signal
a = 1.5; % magnitude
theta = [-pi/2,-pi/2]; % set phase offset
x1 = a*cos(2*pi*f*t + theta(1)); % first channel, first condition
xnull = 0*t; % first channel, second condition

figure('Position',[1 57 1440 748]);
subplot(3,4,1);
shadedErrorBar(t,x1,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-2,2]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
subplot(3,4,2);
stem(f,a,'LineWidth',2,'Color','Blue')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');

subplot(3,4,5);
a2 = a*0.5;
x2 = a2*cos(2*pi*f*t + theta(2)); % second channel, first condition
xnull = xnull; % second channel, second condition (same as first channel)
shadedErrorBar(t,x2,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-2,2]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
subplot(3,4,6);
stem(f,a2,'LineWidth',2,'Color','Blue')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');

subplot(3,4,9);
% compute MI using formulas in paper:
A = diag([a,a2]);

Sig = s.^2*eye(2); % noise covariance matrix
c_om = 0.5*trace(A*inv(Sig)*A*cos(theta-theta'));
temp1 = 0.5*trace(A*inv(Sig)*A*sin(theta+theta'));
temp2 = 0.5*trace(A*inv(Sig)*A*cos(theta+theta'));
r_om = sqrt(temp1.^2 +temp2.^2);
psi_om = atan2(temp1,temp2);
alphaterm = c_om + r_om*cos(2*pi*2*f*t + psi_om);

% and plot these terms:
plot(t,alphaterm,'LineWidth',2,'Color','Black');
xlim([0,max(t)]);
axis square;
grid minor;
plot4paper('Time (sec)','f^{-1}(I_{X,Y})');

subplot(3,4,10);
a3 = c_om;
stem(2*f,a3,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');

%%%%%%%% NOW implement Example 2 on right hand side of figure:

% This example has multiple frequncy components:
f2 = 1.5*f;
a(1) = 0.7*0.7;
a(2) = 1.1*0.7;
x1 = a(1)*sin(2*pi*f*t) + a(2)*sin(2*pi*f2*t); % first channel, first condition

anull = 0.9*[0.2*a(1),0.6];
xnull = anull(1)*sin(2*pi*f*t) + anull(2)*sin(2*pi*f2*t); % first channel, second condition
s = 0.5;
subplot(3,4,3);
shadedErrorBar(t,x1,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-2,2]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
subplot(3,4,4);
stem([f,f2],[a],'LineWidth',2,'Color','Blue'); hold on;
stem([f,f2],anull,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');

subplot(3,4,7);
a2(1) = 0.5*a(1);
a2(2) = 1.5;
x2 = a2(1)*sin(2*pi*f*t) + a2(2)*sin(2*pi*f2*t); % second channel, first condition
xnull = xnull; % second channel, second condition (same as first channel)
shadedErrorBar(t,x2,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-2,2]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
subplot(3,4,8);

%shadedErrorBar(t,x1,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
stem([f,f2],abs([a2]),'LineWidth',2,'Color','Blue');hold on;
stem([f,f2],anull,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');
rng(1);
subplot(3,4,11);

rng(1);
Sigma = Sig;
for i=1:length(t)
    alphaterm(i) = [(x1(i)-xnull(i));(x2(i)-xnull(i))]'*inv(Sigma)*[(x1(i)-xnull(i));(x2(i)-xnull(i))];
end

plot(t,alphaterm,'LineWidth',2,'Color','Black');
xlim([0,max(t)]);
axis square;
grid minor;
plot4paper('Time (sec)','f^{-1}(I_{X,Y})');

subplot(3,4,12);
a3 = 0.6;
stem([2*f,2*f2,f2-f,abs(f+f2)],[1.5,1,0.3,0.4],'LineWidth',2,'Color','Black')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');
print([figdir,'Fig2_InfoSpectrum'],'-dpng');


%% Figure 3: Representational Aliasing

figure('Position',[384 186 840 326]);
clear a;
f = 20;
t = 0.001:0.001:0.1;
x2 = sin(2*pi*f*t);
for i=1:2
    a(i) = subplot(2,2,2*(i-1)+1);
    plot(t,x2,'LineWidth',2,'Color','blue');
    ylim([-1.0001,1.0001]);
    set(gca,'YGrid','on');
    plot4paper('Time (sec)','Magnitude');
    xlim([0,0.1]);
    set(gca,'XTick',0:0.05:0.1);
    set(gca,'YTick',[]);
end

% overlay sampling at less than Nyquist freq/2
axes(a(1))
hold on;
stepsize = 6; 
fsample = 1/(6*diff(t(1:2))); % corresponds to sampling frequency >160Hz
plot(t(5:stepsize:end),x2(5:stepsize:end),'.','LineWidth',2,'Color','black','MarkerSize',20);
xfill = interp(x2(5:stepsize:end),100);
tfill = interp(t(5:stepsize:end),100);
plot(tfill,xfill./max(x2(:,[10])),'LineWidth',2,'Color','black','LineStyle',':');

axes(a(2))

hold on;
sampleperiod = t(end)/(t(end)*30);
stepsize = round(sampleperiod/(t(2)-t(1)));
plot(t(5:stepsize:end),x2(5:stepsize:end),'.','LineWidth',2,'Color','black','MarkerSize',20);

% and compute sine interpolation:
fsample = 1./(stepsize.*(t(2)-t(1))); % this corresponds to 30Hz sampling rate
fNyq = fsample/2;
f_recovered = fNyq - abs(fNyq-f);
x3 = sin(2*pi*f_recovered*t+0.7*pi);
plot(t,x3,'LineWidth',2,'Color','black','LineStyle',':');

a(3) = subplot(2,2,2);
stem(f,1,'blue','LineWidth',2);hold on;
stem(f,1,'k','LineWidth',2,'LineStyle',':');
xlim([0,30]);
ylim([0,1.2]);
set(gca,'XTick',[0:10:40]);
set(gca,'YTick',[]);
a(3).Position([2,4]) = a(1).Position([2,4]);
plot4paper('Frequency (Hz)','PSD');

a(4) = subplot(2,2,4);
stem(f,1,'blue','LineWidth',2);
hold on;
plot4paper('Frequency (Hz)','PSD');
f_recovered = round(f_recovered);
stem(f_recovered,1,'k','LineWidth',2,'LineStyle',':');
xlim([0,30]);
ylim([0,1.2]);
set(gca,'YTick',[]);
a(4).Position([2,4]) = a(2).Position([2,4]);

print([figdir,'Fig3a_Aliasing'],'-dpng');

legend('True frequency','Recovered frequency');

axes(a(1));
legend('True signal','Recovered Signal');
print([figdir,'Fig3b_Aliasing_leg'],'-dpng');

%% Figure 4: Motivation for complex spectrum decoding:

% generate phasor diagrams:

% unit circle:
phi_phasor = 0:0.001:2*pi;
unitcircle = exp(sqrt(-1)*phi_phasor);
figure();plot(unitcircle,'k');
axis square;grid on;
hold on;
theta = pi/2.5;
plot([0,exp(sqrt(-1)*theta)],'LineWidth',2,'Color','black');
plot([exp(sqrt(-1)*theta)],'o','LineWidth',2,'Color','black');
plot([real(exp(sqrt(-1)*theta)),exp(sqrt(-1)*theta)],'Color','black');

x_m = real(exp(sqrt(-1)*theta))+0.005*sqrt(-1);
s = 0.4;
plot(x_m,'*k','LineWidth',2);
plot([x_m - s,x_m+s],'k','LineWidth',1.5);
plot([x_m - s,x_m+s],'k+','LineWidth',1.5);
plot4paper('Real plane','Imaginary plane')

theta = theta + pi;
plot([0,exp(sqrt(-1)*theta)],'LineWidth',2,'Color',[0.3 0.5 0.9]);
plot([exp(sqrt(-1)*theta)],'o','LineWidth',2,'Color',[0.3 0.5 0.9]);
plot([real(exp(sqrt(-1)*theta)),exp(sqrt(-1)*theta)],'Color',[0.3 0.5 0.9]);

x_m = real(exp(sqrt(-1)*theta))-0.005*sqrt(-1);
s = 0.4;
plot(x_m,'*','LineWidth',2,'Color',[0.3 0.5 0.9]);
plot([x_m - s,x_m+s],'','LineWidth',1.5,'Color',[0.3 0.5 0.9]);
plot([x_m - s,x_m+s],'+','LineWidth',1.5,'Color',[0.3 0.5 0.9]);
print([figdir,'Fig4a_RealComplexPlane'],'-dpng');

% plot histograms:
s = s;%/1.5; % arbitrarily rescale for clearer demo
figure('Position',[498 156 452 148]);
x = -3:0.01:3;
y = normpdf(x,real(exp(sqrt(-1)*theta)),s);
h = fill(x,y,[0.3 0.5 0.9]);hold on;
set(h,'facealpha',.5)
y2 = normpdf(x,real(exp(sqrt(-1)*(theta+pi))),s);
h = fill(x,y2,[0 0 0]);
set(h,'facealpha',.5)
xlim([-1,1]);
plot4paper('','');
set(gca,'YTick','');
print([figdir,'Fig4b_RealComplexPlane_hist1'],'-dpng');

figure('Position',[498 156 452 148]);
x = -3:0.01:3;
y = normpdf(x,-1,s);
h = fill(x,y,[0.3 0.5 0.9]);hold on;
set(h,'facealpha',.5)
y2 = normpdf(x,1,s);
h = fill(x,y2,[0 0 0]);
set(h,'facealpha',.5)
xlim([-1,1]);
plot4paper('','');
set(gca,'YTick','');
print([figdir,'Fig4c_RealComplexPlane_hist2'],'-dpng');

%% Figure 5: Simulating signals with non-stationary and non-oscillatory components


% Example 1: non stationary signal based on 'chirp' response functions:
t = 0.001:0.001:0.1;
xchirp = 0.75*chirp(t,50,0.1,2,'logarithmic',270);

t = 0.001:0.001:0.3;

Sigma = Sig; % noise covariance
x1 = [zeros(1,100),xchirp,zeros(1,100)];
xnull = 0*t;
s = 0.5;
figure('Position',[1 57 1440 748]);
subplot(3,6,1);

shadedErrorBar(t,x1,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-1,1]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');


subplot(3,6,7);
x2 = [zeros(1,150),xchirp,zeros(1,50)]; % second channel: chirp with different onset time
shadedErrorBar(t,x2,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-1,1]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain - using STFT for estimation
subplot(3,6,2);
W(1) = 50; % we will use STFTs with two different window sizes
W(2) = 250;
offset = [1,7]; % this just for subplot references

rng(1)
for i=1:2 % iterate over STFT window size parameters
    clear Pxx Ixy
    for k=1:2 % iterate over channels
        subplot(3,6,offset(k)+i)
        if k==1
            input = [zeros(1,W(i)),x1,zeros(1,W(i))];
        else
            input = [zeros(1,W(i)),x2,zeros(1,W(i))];
        end
        % run several sims for noise and average:
        clear Ptemp1
        for isim=1:10
            input1 = input + 0.25*randn(size(input));
            [Ptemp1(:,:,isim),F,T] = spectrogram(input1,W(i),W(i)-1,[1:1:100],1/diff(t(1:2)));
        end
        Ptemp = mean(Ptemp1,3);
        Pxx(:,:,k) = Ptemp(:,ceil(W(i)/2)+1:(end-ceil(W(i)/2)-1),:);
        imagesc(log(flipud(abs(Pxx(:,:,k)))));
        caxis(log([5,max(abs(squash(Pxx(:,:,k))))]))

        set(gca,'XTick',[1,round(size(Pxx,2)/3),round(2*size(Pxx,2)/3),size(Pxx,2)]);
        set(gca,'XTickLabel',[0,t([100:100:300])]);
        set(gca,'YTick',[1,20:20:100]);
        set(gca,'YTickLabel',flipud([0;F([20:20:100])]));
        plot4paper('Time (sec)','Freq (Hz)');
        axis square;
        
    end

    % plot the MI terms in each frequency band vs time:
    subplot(3,6,13+i);
    
    for it=1:length(t)
        acc(it) = [(x1(it)-xnull(it));(x2(it)-xnull(it))]'*inv(Sigma)*[(x1(it)-xnull(it));(x2(it)-xnull(it))];
    end
    input = [zeros(1,W(i)),acc,zeros(1,W(i))];
    [Ptemp,F,T] = spectrogram(input,W(i),W(i)-1,[1:1:100],1/diff(t(1:2)));
    Pxx_acc = Ptemp(:,ceil(W(i)/2)+1:(end-ceil(W(i)/2)-1),:);
    imagesc(log(flipud(abs(Pxx_acc(:,:)))));
    caxis(log([5,max(abs(Pxx_acc(:)))]))

    set(gca,'XTick',[1,round(size(Pxx_acc,2)/3),round(2*size(Pxx_acc,2)/3),size(Pxx_acc,2)]);
    set(gca,'XTickLabel',[0,t([100:100:300])]);
    set(gca,'YTick',[1,20:20:100]);
    set(gca,'YTickLabel',flipud([0;F([20:20:100])]));
    plot4paper('Time (sec)','Freq (Hz)');
    axis square;
end

% plot the MI term of the broadband signal vs time:
subplot(3,6,13);
for i=1:length(t)
    acc(i) = [(x1(i)-xnull(i));(x2(i)-xnull(i))]'*inv(Sigma)*[(x1(i)-xnull(i));(x2(i)-xnull(i))];
end
plot(t,acc,'LineWidth',2,'Color','Black');
xlim([0,max(t)]);
axis square;
grid minor;
plot4paper('Time (sec)','f^{-1}(I_{X,Y})');

% Example 2: we now simulate a signal comrpising two discrete 'non
% sinusoidal' responses, using gaussian kernel functions to simulate
% 'activations' that are not oscillatory in nature:

x1 = zeros(size(t));
x1(100) = 1; % impulse function
kernelfunc = 0.75*exp(-0.75*[-3:0.1:3].^2); % gaussian kernel
x1 = conv(x1,kernelfunc,'same');

s = Sigma(1);
subplot(3,6,4);

shadedErrorBar(t,x1,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-1,1]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');



subplot(3,6,10);
x2 = zeros(size(t));
x2(200) = 1;
x2 = conv(x2,kernelfunc,'same');
shadedErrorBar(t,x2,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-1,1]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
offset = [4,10];
for i=1:2  % iterate over the two STFT window size parameters
    clear Pxx Ixy 
    for k=1:2 % iterate over the two channels
        subplot(3,6,offset(k)+i)
        if k==1
            input = [zeros(1,W(i)),x1,zeros(1,W(i))];
        else
            input = [zeros(1,W(i)),x2,zeros(1,W(i))];
        end
        input = input + 0.01*randn(size(input));
        [Ptemp,F,T] = spectrogram(input,W(i),W(i)-1,[1:1:100],1/diff(t(1:2)));
        Pxx(:,:,k) = Ptemp(:,ceil(W(i)/2)+1:(end-ceil(W(i)/2)-1),:);
        % plot the channel time-frequency responses:
        imagesc(log(flipud(abs(Pxx(:,:,k)))));
        caxis(log([5,max(abs(squash(Pxx(:,:,k))))]))
        set(gca,'XTick',[1,round(size(Pxx,2)/3),round(2*size(Pxx,2)/3),size(Pxx,2)]);
        set(gca,'XTickLabel',[0,t([100:100:300])]);
        set(gca,'YTick',[1,20:20:100]);
        set(gca,'YTickLabel',flipud([0;F([20:20:100])]));
        plot4paper('Time (sec)','Freq (Hz)');
        axis square;
    end
    % plot the MI terms in each frequency band vs time:
    subplot(3,6,16+i);
    for it=1:length(t)
        acc(it) = [(x1(it)-xnull(it));(x2(it)-xnull(it))]'*inv(Sigma)*[(x1(it)-xnull(it));(x2(it)-xnull(it))];
    end
    input = [zeros(1,W(i)),acc,zeros(1,W(i))];
    [Ptemp,F,T] = spectrogram(input,W(i),W(i)-1,[1:1:100],1/diff(t(1:2)));
    Pxx_acc = Ptemp(:,ceil(W(i)/2)+1:(end-ceil(W(i)/2)-1),:);
    imagesc(log(flipud(abs(Pxx_acc(:,:)))));
    caxis(log([5,max(abs(Pxx_acc(:)))]))

    set(gca,'XTick',[1,round(size(Pxx_acc,2)/3),round(2*size(Pxx_acc,2)/3),size(Pxx_acc,2)]);
    set(gca,'XTickLabel',[0,t([100:100:300])]);
    set(gca,'YTick',[1,20:20:100]);
    set(gca,'YTickLabel',flipud([0;F([20:20:100])]));
    plot4paper('Time (sec)','Freq (Hz)');
    axis square;
end

% plot the MI term of the broadband signal vs time:
subplot(3,6,16)
for i=1:length(t)
    acc(i) = [(x1(i)-xnull(i));(x2(i)-xnull(i))]'*inv(Sigma)*[(x1(i)-xnull(i));(x2(i)-xnull(i))];
end
plot(t,acc,'LineWidth',2,'Color','Black');
hold on;
xlim([0,max(t)]);

axis square;
grid minor;
colormap hot;
plot4paper('Time (sec)','f^{-1}(I_{X,Y})');

print([figdir,'Fig5_InfoSpectrum'],'-dpng');