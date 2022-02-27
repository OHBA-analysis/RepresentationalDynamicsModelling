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


%% also make gif files:

h = figure('Position',[440 462 2*285 336],'Color','white');
giffilename = [figdir,'Fig1_pendulum1.gif'];
for it = 1:2:length(t)
    clear l
    ax1 = subplot(2,2,2);
    plot(t,x,'k','LineWidth',2);
    plot4paper('Time','Displacement');
    hold on;
    l(1) = plot(t(it),x(it),'k.','MarkerSize',40);
    ylim([-1.2,1.2])
    %hold off;
    %plot(t,x_dash,'r','LineWidth',2,'LineStyle','--');
    %hold on;
    %l(2) = plot(t(it),x_dash(it),'r.','MarkerSize',40);
    
    %legend('{\it x}')
    clear l
    l(1) = plot(nan,nan,'k.-','LineWidth',2,'MarkerSize',40);
    %l(2) = plot(nan,nan,'r.--','LineWidth',2,'MarkerSize',40);
    hold off;
    leg1 = legend(l,{'{$x$}'},'Interpreter','latex')
    %leg = legend(l,{'I[{$x$}]','I[$x,\dot{x}$]'}, 'Interpreter','latex');
    %leg1 = legend(l,{'1','2'},'Interpreter','latex')
    set(leg1,'Position',[0.8858 0.7198 0.0864 0.1162])
    %legend('$\dot{x}$', 'Interpreter','latex')
    % and plot information content associated with this:
    sigma = 0.5;
    for i=1:length(t)
        alphaterm(i,1) = 2*x(i).^2/sigma;
        alphaterm(i,2) = 2*[x(i),x_dash(i)]*inv(diag([sigma,sigma]))*[x(i),x_dash(i)]';
    end
    clear l;
    set(ax1,'Position',[0.5 0.6301 0.35 0.2949])
    subplot(2,2,4);
    plot(t,alphaterm(:,1),'k','LineWidth',2);
    plot4paper('Time','Info Content');
    set(gca,'YTick','')
    ylim([0,alphaterm(1,2)*1.2]);
    hold on;
    plot(t(it),alphaterm(it,1),'k.','MarkerSize',40);
    
    
    %plot(t,alphaterm(:,2),'r','LineWidth',2,'LineStyle','--');
    %plot(t(it),alphaterm(it,2),'r.','MarkerSize',40);
    
    l(1) = plot(nan,nan,'k.-','LineWidth',2,'MarkerSize',40);
    %l(2) = plot(nan,nan,'r.--','LineWidth',2,'MarkerSize',40);
    hold off;
    %legend(l,{'I{x}','I(x,)'},'Interpreter','latex','Location','SouthWest')
    leg = legend(l,{'I[{$x$}]'}, 'Interpreter','latex');
    set(leg,'Position', [0.8586 0.2124 0.1352 0.1162])
    set(gca,'Position',[0.503 0.13063 0.35 0.2949])
    % plot actual pendulum:
    subplot(1,2,1)
    %axis off
    
    r = 0.5;
    offset(1) = cos(r*x(it) - pi/2);
    offset(2) = sin(r*x(it) - pi/2);
    plot([0,offset(1)],[0,offset(2)],'k','LineWidth',2)
    %axis off
    ax=get(gca);
    
    ax.XAxis.Label.Color=[0 0 0];
    ax.XAxis.Label.Visible='on';
    ax.XRuler.Axle.Visible = 'off'; % ax is axis handle
    ax.YRuler.Axle.Visible = 'off'; 
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    set(gca,'Visible','off')
    hold on;
    plot(offset(1),offset(2),'k.','MarkerSize',40)
    hold off;
    xlim(0.5*[-1,1])
    ylim([-1,0])
    axis square
    labeltext = '$\begin{array}{c}{\it x} =$ Displacement$\\\end{array}$';
    xlabel(labeltext, 'Interpreter','latex','FontSize',20)

    % Write to the GIF File 
    frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      if it == 1 
          imwrite(imind,cm,giffilename,'gif', 'DelayTime',0.1,'Loopcount',inf); 
      else 
          imwrite(imind,cm,giffilename,'gif','WriteMode','append','DelayTime',0.07); 
      end 
end
%
h = figure('Position',[440 462 2*285 336],'Color','white');
giffilename = [figdir,'Fig1_pendulum2.gif'];
for it = 1:2:length(t)
    clear l
    ax1 = subplot(2,2,2);
    plot(t,x,'k','LineWidth',2);
    plot4paper('Time','Displacement');
    hold on;
    l(1) = plot(t(it),x(it),'k.','MarkerSize',40);
    ylim([-1.2,1.2])
    %hold off;
    plot(t,x_dash,'r','LineWidth',2,'LineStyle','--');
    %hold on;
    l(2) = plot(t(it),x_dash(it),'r.','MarkerSize',40);
    
    %legend('{\it x}')
    clear l
    l(1) = plot(nan,nan,'k.-','LineWidth',2,'MarkerSize',40);
    l(2) = plot(nan,nan,'r.--','LineWidth',2,'MarkerSize',40);
    hold off;
    leg1 = legend(l,{'{$x$}','$\dot{x}$'},'Interpreter','latex')
    %leg = legend(l,{'I[{$x$}]','I[$x,\dot{x}$]'}, 'Interpreter','latex');
    %leg1 = legend(l,{'1','2'},'Interpreter','latex')
    set(leg1,'Position',[0.8858 0.7198 0.0864 0.1162])
    %legend('$\dot{x}$', 'Interpreter','latex')
    % and plot information content associated with this:
    sigma = 0.5;
    for i=1:length(t)
        alphaterm(i,1) = 2*x(i).^2/sigma;
        alphaterm(i,2) = 2*[x(i),x_dash(i)]*inv(diag([sigma,sigma]))*[x(i),x_dash(i)]';
    end
    clear l;
    set(ax1,'Position',[0.5 0.6301 0.35 0.2949])
    subplot(2,2,4);
    plot(t,alphaterm(:,1),'k','LineWidth',2);
    plot4paper('Time','Info Content');
    set(gca,'YTick','')
    ylim([0,alphaterm(1,2)*1.2]);
    hold on;
    plot(t(it),alphaterm(it,1),'k.','MarkerSize',40);
    
    
    plot(t,alphaterm(:,2),'r','LineWidth',2,'LineStyle','--');
    plot(t(it),alphaterm(it,2),'r.','MarkerSize',40);
    
    l(1) = plot(nan,nan,'k.-','LineWidth',2,'MarkerSize',40);
    l(2) = plot(nan,nan,'r.--','LineWidth',2,'MarkerSize',40);
    hold off;
    %legend(l,{'I{x}','I(x,)'},'Interpreter','latex','Location','SouthWest')
    leg = legend(l,{'I[{$x$}]','I[$x,\dot{x}$]'}, 'Interpreter','latex');
    set(leg,'Position', [0.8586 0.2124 0.1352 0.1162])
    set(gca,'Position',[0.503 0.13063 0.35 0.2949])
    % plot actual pendulum:
    subplot(1,2,1)
    %axis off
    
    r = 0.5;
    offset(1) = cos(r*x(it) - pi/2);
    offset(2) = sin(r*x(it) - pi/2);
    plot([0,offset(1)],[0,offset(2)],'k','LineWidth',2)
    %axis off
    ax=get(gca);
    
    ax.XAxis.Label.Color=[0 0 0];
    ax.XAxis.Label.Visible='on';
    ax.XRuler.Axle.Visible = 'off'; % ax is axis handle
    ax.YRuler.Axle.Visible = 'off'; 
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    set(gca,'Visible','off')
    hold on;
    plot(offset(1),offset(2),'k.','MarkerSize',40)
    hold off;
    xlim(0.5*[-1,1])
    ylim([-1,0])
    axis square
    labeltext = '$\begin{array}{c}{\it x} =$ Displacement$\\{\dot{x}} =$ Velocity$\end{array}$';
    xlabel(labeltext, 'Interpreter','latex','FontSize',20)

    % Write to the GIF File 
    frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      if it == 1 
          imwrite(imind,cm,giffilename,'gif', 'DelayTime',0.1,'Loopcount',inf); 
      else 
          imwrite(imind,cm,giffilename,'gif','WriteMode','append','DelayTime',0.07); 
      end 
end

%% Figure 2: 

% Simulate the examples shown in figure 2:

% Example 1:
t = 0.001:0.001:0.1;
s = 0.5; % noise standard deviation
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
A = 0.5*diag([a,a2]);

Sig = s.^2*eye(2); % noise covariance matrix
c_om = trace(A*inv(Sig)*A*cos(theta-theta'));
temp1 = trace(A*inv(Sig)*A*sin(theta+theta'));
temp2 = trace(A*inv(Sig)*A*cos(theta+theta'));
r_om = sqrt(temp1.^2 +temp2.^2);
psi_om = atan2(temp1,temp2);
alphaterm = c_om + r_om*cos(2*pi*2*f*t + psi_om);

% sanity check:
clear alphaterm2
for i=1:length(t)
    alphaterm2(i) = 2*[(x1(i)-xnull(i))/2;(x2(i)-xnull(i))/2]'*inv(Sig)*[(x1(i)-xnull(i))/2;(x2(i)-xnull(i))/2];
end

% and plot these terms:
plot(t,alphaterm,'LineWidth',2,'Color','Black');
hold on;
plot(t,alphaterm2,'LineWidth',2,'Color','Black');
xlim([0,max(t)]);
axis square;
grid minor;
plot4paper('Time (sec)','f^{-1}(I_{X,Y})');

subplot(3,4,10);
a3 = c_om;
stem(2*f,a3,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
%ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');

%%%%%%%% NOW implement Example 2 on right hand side of figure:

% This example has multiple frequncy components:
f2 = 1.5*f;
freqs = [f;f2];
a(1) = 0.5;
a(2) = 0.77;
phi = [-pi/2;-pi/2]; % set phase offset
x1 = a*cos(2*pi*freqs*t + repmat(phi,1,length(t))); % first channel, first condition

anull = 0.9*[0.2*a(1),0.6];
xnull = anull*sin(2*pi*freqs*t); % first channel, second condition

s = 0.5; % set standard deviation of noise on each channel
subplot(3,4,3);
shadedErrorBar(t,x1,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-2,2]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
subplot(3,4,4);
stem(freqs,[a],'LineWidth',2,'Color','Blue'); hold on;
stem(freqs,anull,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');

subplot(3,4,7);
a2(1) = 0.5*a(1);
a2(2) = 1.5;
x2 = a2 * sin(2*pi*freqs*t); % second channel, first condition
anull_2 = anull;
xnull_2 = anull_2*sin(2*pi*freqs*t); % second channel, second condition (same as first channel)

shadedErrorBar(t,x2,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull_2,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7)
ylim([-2,2]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
subplot(3,4,8);
stem(freqs,a2,'LineWidth',2,'Color','Blue');hold on;
stem(freqs,anull_2,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');

% now we need to map these two channel signals back to the format given in
% the paper:
for ifreq=1:2
    temp1 = [a(ifreq),a2(ifreq)];
    temp2 = [anull(ifreq),anull_2(ifreq)];
    temp3 = [-pi/2,-pi/2];
    temp4 = [-pi/2,-pi/2];
    A_omega{ifreq} = 0.5*diag(sqrt(temp1.^2+temp2.^2 + ...
        2*temp1.*temp2.*cos(temp3 - temp4 + [-pi,-pi])));
    mu_omega{ifreq} = 0.5*diag(sqrt(temp1.^2+temp2.^2 + ...
        2*temp1.*temp2.*cos(temp3 - temp4)));
    Phi_omega{ifreq} = atan2(temp1.*sin(temp3) + temp2.*sin(temp4 + pi),temp1.*cos(temp3) + temp2.*cos(temp4 + pi))';
    Phi_mean{ifreq} = -pi/2 + atan2(temp1.*sin(temp3 + pi/2) + temp2.*sin(temp4 + pi/2),temp1.*cos(temp3 + pi/2) + temp2.*cos(temp4 + pi/2))';
end

% find the information content terms:
c_b = 0;
for ifreq=1:2
    c_b = c_b + trace(A_omega{ifreq}*inv(Sigma)*A_omega{ifreq}*cos(Phi_omega{ifreq} - Phi_omega{ifreq}')); % equal to above expression
    temp1 = trace(A_omega{ifreq}*inv(Sigma)*A_omega{ifreq}*cos(Phi_omega{ifreq} + Phi_omega{ifreq}'));
    temp2 = trace(A_omega{ifreq}*inv(Sigma)*A_omega{ifreq}*sin(Phi_omega{ifreq} + Phi_omega{ifreq}'));
    r_b(ifreq) = sqrt(temp1.^2 + temp2.^2)
    psi(ifreq) = atan2(temp2,temp1);
end

% and cross-frequency components:
ifreq1 = 1; ifreq2 = 2;

temp1 = trace(A_omega{ifreq1}*inv(Sigma)*A_omega{ifreq2}*cos(Phi_omega{ifreq1} + Phi_omega{ifreq2}'));
temp2 = trace(A_omega{ifreq1}*inv(Sigma)*A_omega{ifreq2}*sin(Phi_omega{ifreq1} + Phi_omega{ifreq2}'));
r_b(3) = 2*sqrt(temp1.^2 + temp2.^2);
psi(3) = atan2(temp2,temp1);


temp1 = trace(A_omega{ifreq1}*inv(Sigma)*A_omega{ifreq2}*cos(Phi_omega{ifreq1} - Phi_omega{ifreq2}'));
temp2 = trace(A_omega{ifreq1}*inv(Sigma)*A_omega{ifreq2}*sin(Phi_omega{ifreq2} - Phi_omega{ifreq1}'))
r_b(4) = 2*sqrt(temp1.^2 + temp2.^2);
psi(4) = atan2(temp2,temp1);

infotermest = c_b;
freqs_all = [2*freqs;sum(freqs);diff(freqs)];
for ifreq=1:4
    infotermest = infotermest + r_b(ifreq)*cos(2*pi*freqs_all(ifreq)*t + psi(ifreq));
end

subplot(3,4,11);
Sigma = Sig + Sig; % broadband noise: sum over both frequency bands
% note - as a sanity check, infotermest should be equal to alpha term computed as follows:
for i=1:length(t)
    alphaterm(i) = 2*[(x1(i)-xnull(i))/2;(x2(i)-xnull_2(i))/2]'*inv(Sigma)*[(x1(i)-xnull(i))/2;(x2(i)-xnull_2(i))/2];
end

plot(t,alphaterm,'LineWidth',2,'Color','Black');

%analytical version:
infotermest = c_b;
for ifreq=1:4
    infotermest = infotermest + r_b(ifreq)*cos(2*pi*freqs_all(ifreq)*t + psi(ifreq))
end
freqs_all = [2*freqs;sum(freqs);diff(freqs)];

%sanity check both are the same:
hold on;
plot(t,infotermest,'LineWidth',2,'Color','Black')
xlim([0,max(t)]);
axis square;
grid minor;
plot4paper('Time (sec)','f^{-1}(I_{X,Y})');

subplot(3,4,12);
%stem(freqs_all,[1.5,1,0.3,0.4],'LineWidth',2,'Color','Black')
stem(freqs_all,r_b,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
%ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');
print([figdir,'Fig2_InfoSpectrum'],'-dpng');

%% Online Toolbox;

% this section of the code largely repeats figure 2 above but for arbitrary
% phase, magnitude and frequency parameters

clearvars -except figdir;
% Example 1:
t = 0.001:0.001:0.1;
s = 0.5; % noise magnitude

% let us now specify channel parameters:
figure('Position',[64 55 1240 750]);
% This example has multiple frequncy components:

freqs = (rand(2,1)*20); % set frequency of channel components
a = rand(1,2); % set amplitude on channel one, condition one, over both frequencies
phi1 = rand(2,1)*2*pi; % set phase offset on channel 1, condition 1, over both frequencies
x1 = a*cos(2*pi*freqs*t + repmat(phi1,1,length(t))); % first channel, first condition

% and specify null condition on same channel:
anull = rand(1,2); % set amplitude on channel one, null condition, over both frequencies
phi1_null = rand(2,1)*2*pi; % set phase offset on channel 1, null condition, over both frequencies
xnull = anull*cos(2*pi*freqs*t + repmat(phi1_null,1,length(t))); % first channel, second condition

s = 0.5; % set standard deviation
Sig = s.^2*eye(2); % set channel covariance matrix

subplot(3,4,3);
shadedErrorBar(t,x1,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7);
ylim([-2,2]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
subplot(3,4,4);
stem(freqs,[a],'LineWidth',2,'Color','Blue'); hold on;
stem(freqs,anull,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');



% now, let us simulate a second channel for the same two conditions and
% frequncies:
a2 = rand(1,2); % set amplitude on channel two, condition one, over both frequencies
phi2 = rand(2,1)*2*pi;[-pi/2;-pi/2]; % set phase offset on channel 2, condition 1, over both frequencies
x2 = a2 * cos(2*pi*freqs*t + repmat(phi2,1,length(t))); % second channel, first condition

anull_2 = rand(1,2); % set amplitude on channel two, null condition, over both frequencies
phi2_null = rand(2,1)*2*pi; % set phase offset on channel two, null condition, over both frequencies
xnull_2 = anull_2*cos(2*pi*freqs*t + repmat(phi2_null,1,length(t))); % data for second channel, second condition

% plot these signals:
subplot(3,4,7);
shadedErrorBar(t,x2,s*ones(length(t),1),{'LineWidth',2,'Color','Blue'},0.7);hold on;
shadedErrorBar(t,xnull_2,s*ones(length(t),1),{'LineWidth',2,'Color','Black'},0.7);
ylim([-2,2]);
axis square;
grid minor;
plot4paper('Time (sec)','Magnitude');

% and plot in Fourier Domain:
subplot(3,4,8);
stem(freqs,a2,'LineWidth',2,'Color','Blue');hold on;
stem(freqs,anull_2,'LineWidth',2,'Color','Black')
xlim([0,40.5]);
ylim([0,2]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');


% now plot experimentally computed alpha term (this is a sanity check, we compute it
% analytically below):
subplot(3,4,11);
Sigma = Sig + Sig; % broadband noise: sum over frequency bands
for i=1:length(t)
    alphaterm(i) = 2*[(x1(i)-xnull(i))/2;(x2(i)-xnull_2(i))/2]'*inv(Sigma)*[(x1(i)-xnull(i))/2;(x2(i)-xnull_2(i))/2];
end

plot(t,alphaterm,'LineWidth',2,'Color','Black');

% now convert these to the variable names in the paper to obtain analytical frequency spectra:
for ifreq=1:2
    temp1 = [a(ifreq),a2(ifreq)];
    temp2 = [anull(ifreq),anull_2(ifreq)];
    temp3 = [phi1(ifreq),phi2(ifreq)];
    temp4 = [phi1_null(ifreq),phi2_null(ifreq)];
    A_omega{ifreq} = 0.5*diag(sqrt(temp1.^2+temp2.^2 + ...
        2*temp1.*temp2.*cos(temp3 - temp4 + [-pi,-pi])));
    mu_omega{ifreq} = 0.5*diag(sqrt(temp1.^2+temp2.^2 + ...
        2*temp1.*temp2.*cos(temp3 - temp4)));
    Phi_omega{ifreq} = atan2(temp1.*sin(temp3) + temp2.*sin(temp4 + pi),temp1.*cos(temp3) + temp2.*cos(temp4 + pi))';
    Phi_mean{ifreq} = -pi/2 + atan2(temp1.*sin(temp3 + pi/2) + temp2.*sin(temp4 + pi/2),temp1.*cos(temp3 + pi/2) + temp2.*cos(temp4 + pi/2))';
end

% find the information content terms:
c_b = 0;
for ifreq=1:2
    c_b = c_b + trace(A_omega{ifreq}*inv(Sigma)*A_omega{ifreq}*cos(Phi_omega{ifreq} - Phi_omega{ifreq}')); % equal to above expression
    temp1 = trace(A_omega{ifreq}*inv(Sigma)*A_omega{ifreq}*cos(Phi_omega{ifreq} + Phi_omega{ifreq}'));
    temp2 = trace(A_omega{ifreq}*inv(Sigma)*A_omega{ifreq}*sin(Phi_omega{ifreq} + Phi_omega{ifreq}'))
    r_b(ifreq) = sqrt(temp1.^2 + temp2.^2)
    psi(ifreq) = atan2(temp2,temp1);
end

% and cross-frequency components:
ifreq1 = 1; ifreq2 = 2;

temp1 = trace(A_omega{ifreq1}*inv(Sigma)*A_omega{ifreq2}*cos(Phi_omega{ifreq1} + Phi_omega{ifreq2}'));
temp2 = trace(A_omega{ifreq1}*inv(Sigma)*A_omega{ifreq2}*sin(Phi_omega{ifreq1} + Phi_omega{ifreq2}'));
r_b(3) = 2*sqrt(temp1.^2 + temp2.^2);
psi(3) = atan2(temp2,temp1);


temp1 = trace(A_omega{ifreq1}*inv(Sigma)*A_omega{ifreq2}*cos(Phi_omega{ifreq1} - Phi_omega{ifreq2}'));
temp2 = trace(A_omega{ifreq1}*inv(Sigma)*A_omega{ifreq2}*sin(Phi_omega{ifreq2} - Phi_omega{ifreq1}'))
r_b(4) = 2*sqrt(temp1.^2 + temp2.^2);
psi(4) = atan2(temp2,temp1);

%sanity check:
infotermest = c_b;
freqs_all = [2*freqs;sum(freqs);diff(freqs)];
for ifreq=1:4
    infotermest = infotermest + r_b(ifreq)*cos(2*pi*freqs_all(ifreq)*t + psi(ifreq));
    
end


%sanity check both are the same (this line should be identical to the one already plotted):
hold on;
plot(t,infotermest,'LineWidth',2,'Color','Red');
xlim([0,max(t)]);
axis square;
grid minor;
plot4paper('Time (sec)','f^{-1}(I_{X,Y})');


subplot(3,4,12);
stem(abs(freqs_all),r_b,'LineWidth',2,'Color','Black');
xlim([0,40.5]);
axis square;
grid minor;
plot4paper('Freq (Hz)','PSD');


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

x1 = [zeros(1,100),xchirp,zeros(1,100)];
xnull = 0*t;
s = 0.25;
Sigma = s.^2*eye(2); % noise covariance
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
        acc(it) = 0.5*[(x1(it)-xnull(it));(x2(it)-xnull(it))]'*inv(Sigma)*[(x1(it)-xnull(it));(x2(it)-xnull(it))];
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
    acc(i) = 0.5*[(x1(i)-xnull(i));(x2(i)-xnull(i))]'*inv(Sigma)*[(x1(i)-xnull(i));(x2(i)-xnull(i))];
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

s = sqrt(Sigma(1));
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
        acc(it) = 0.5*[(x1(it)-xnull(it));(x2(it)-xnull(it))]'*inv(Sigma)*[(x1(it)-xnull(it));(x2(it)-xnull(it))];
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
    acc(i) = 0.5*[(x1(i)-xnull(i));(x2(i)-xnull(i))]'*inv(Sigma)*[(x1(i)-xnull(i));(x2(i)-xnull(i))];
end
plot(t,acc,'LineWidth',2,'Color','Black');
hold on;
xlim([0,max(t)]);

axis square;
grid minor;
colormap hot;
plot4paper('Time (sec)','f^{-1}(I_{X,Y})');

print([figdir,'Fig5_InfoSpectrum'],'-dpng');