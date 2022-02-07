% this script plots the results of the analysis conducted by the script
% MEGDataAnalysis, to obtain the plots given in Higgins et al 2022.


%% Figure 6A: compare instantaneous, nrrowband and complex spectrum decoding:
DM = [true(30,1);false(30,1)];
DM = repmat(permute(DM,[2,3,1]),7,111);
acc_all = nan(nF,ttrial,ncomparisons,15);
acc_realonly = nan(nF,ttrial,ncomparisons,15);
for iSj=1:15
    fprintf(['Sj=',int2str(iSj),'\n'])
    % load real part:
    %acc_realonly = nan(nF,ttrial,ncomparisons);
    indcomp = 1;
    for icond1 = 1:ncond
        for icond2=(icond1+1):ncond
            load([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\acc_preds',sprintf('%04d',indcomp),'.mat'],'preds');
            %setup data format:
            preds = cast(preds>0,'uint8');
            acc_all(:,:,indcomp,iSj) = mean(~xor(DM,preds),3);
            
            load([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\acc_preds',sprintf('%04d',indcomp),'_realonly.mat'],'preds');
            preds = cast(preds>0,'uint8');
            acc_realonly(:,:,indcomp,iSj) = mean(~xor(DM,preds),3);
            indcomp = indcomp+1;
        end
    end
end
temp_acc_real = permute(mean(acc_realonly,3),[2,1,4,3]);

figure('Position',[357 494 883 604]);
cols = parula(nF);
cols = [0,0,0;cols(1:nF-1,:);0*[1,1,1]];
clear h;
temp_acc_real(setdiff(1:111,t_to_run),:,:) = NaN;
nSj_subset = size(acc_realonly,4);
for iF=1:nF
    shadedErrorBar(t_points,mean(temp_acc_real(:,iF,:),3),std(temp_acc_real(:,iF,:),[],3)./sqrt(nSj_subset),{'Color',cols(iF,:),'LineWidth',2},0.5);
    hold on;
    h(iF) = plot(nan(2,1),nan(2,1),'Color',cols(iF,:),'LineWidth',2);
end
ylim([0.48,0.68]);plot4paper('Time(sec)','Accuracy');
labelsplot{1} = 'Inst.';
for i=2:7
    labelsplot{i} = [int2str(freq_bands(i)),' Hz'];
end
legend(h,labelsplot)
xlim([0,0.5])
title('Real Spectrum Decoding');
print([figdir,'Fig6A_realonly_allsj'],'-dpng');

temp_acc_all = permute(mean(acc_all,3),[2,1,4,3]);
figure('Position',[357 494 883 604]);
cols = parula(nF);
cols = [0,0,0;cols(1:nF-1,:);0*[1,1,1]];
clear h;
temp_acc_all(setdiff(1:111,t_to_run),:,:) = NaN;
nSj_subset = size(acc_realonly,4);
for iF=1:nF
    shadedErrorBar(t_points,mean(temp_acc_all(:,iF,:),3),std(temp_acc_all(:,iF,:),[],3)./sqrt(nSj_subset),{'Color',cols(iF,:),'LineWidth',2},0.5);
    hold on;
    h(iF) = plot(nan(2,1),nan(2,1),'Color',cols(iF,:),'LineWidth',2);
end
ylim([0.48,0.68]);plot4paper('Time(sec)','Accuracy');
labelsplot{1} = 'Wideband';
for i=2:7
    labelsplot{i} = [int2str(freq_bands(i)),' Hz'];
end

legend(h,labelsplot)
xlim([0,0.5])
title('Complex Spectrum decoding');
print([figdir,'Fig6A_realimag_allsj'],'-dpng');

%% FIGURE 6B: Accuracy vs time plots in each frequency band:

temp_acc_all = permute(mean(acc_all,3),[2,1,4,3]);
temp_acc_real = permute(mean(acc_realonly,3),[2,1,4,3]);

figure('Position', [7 694 1911 404]);
for iF=2:nF
    subplot(1,nF-1,iF-1);
    plot(t_points,mean(temp_acc_all(:,iF,:),3),'LineWidth',2,'Color',cols(iF,:));hold on;
    plot(t_points,mean(temp_acc_real(:,iF,:),3),'LineWidth',2,'Color',cols(iF,:),'LineStyle',':');hold on;
    
    corrp = osl_clustertf(permute(temp_acc_all(:,iF,:) - temp_acc_real(:,iF,:),[3,2,1]));
    xlim([0,0.5])
    axis square;
    plot4paper('Time (sec)','Accuracy')
    YL = ylim;
    ylim([0.48,YL(2)]); YL = ylim;
    title(labelsplot{iF});
    hold on;
    p_toplot = [corrp>0.975];
    plot(t_points(find(p_toplot)),(YL(1)+0.035*diff(YL))*ones(sum(p_toplot),1),'LineWidth',4,'Color',[0.4660, 0.6740, 0.1880]);

end

clear h;
h(1) = plot(nan,nan,'LineWidth',2,'Color',cols(1,:));
h(2) = plot(nan,nan,'LineWidth',2,'Color',cols(1,:),'LineStyle',':');
l = legend(h,{'Complex Spectrum','Real spectrum'})
set(l,'Position', [0.4654 0.0095 0.1164 0.1349])

print([figdir,'Fig6b_',int2str(imethod)','Accvstime_long'],'-dpng');

%% FIGURE 6C: Example subject:

% this is the example comparison selected in the paper:
iSj=5;
icond=4063;

% find and reload original:
indcomp = 1;
for icond1 = 1:ncond
    for icond2=(icond1+1):ncond
        if indcomp==icond
            load([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\acc_preds',sprintf('%04d',indcomp),'.mat'],'preds');
            %setup data format:
            preds = cast(preds>0,'uint8');
            %acc_all(:,:,indcomp,iSj) = mean(~xor(DM,preds),3);
            x1 = mean(~xor(DM,preds),3);
            ste1 = std(~xor(DM,preds),[],3)/60;
            load([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\acc_preds',sprintf('%04d',indcomp),'_realonly.mat'],'preds');
            preds = cast(preds>0,'uint8');
            %acc_realonly(:,:,indcomp,iSj) = mean(~xor(DM,preds),3);
            x2 = mean(~xor(DM,preds),3);
            ste2 = std(~xor(DM,preds),[],3)/60;
        end
        indcomp = indcomp+1;
    end
end
figure('Position', [130 518 1791 580]);
for iF = 3:6
    subplot(2,4,iF-2)
    shadedErrorBar(t_points,x1(iF,:),ste1(iF,:),{'Color',cols(iF,:),'LineWidth',2},0.5);hold on;
    shadedErrorBar(t_points,x2(iF,:),ste2(iF,:),{'Color',cols(iF,:),'LineWidth',2,'LineStyle',':'});
    plot4paper('Time (sec)','Accuracy');
    plot([0,0.5],[0.5,0.5],'k--');
    xlim([t_points(t_to_run(1)),t_points(t_to_run(end))])
    title(labelsplot{iF})
    ylim([0.4,0.98]);
    subplot(2,4,4+iF-2)
    temp = pspectrum(acc_all(iF,t_to_run,icond,iSj)-0.5);
    temp2 = pspectrum(acc_realonly(iF,t_to_run,icond,iSj)-0.5);
    temp = pwelch(acc_all(iF,t_to_run,icond,iSj)-0.5,25,20);
    temp2 = pwelch(acc_realonly(iF,t_to_run,icond,iSj)-0.5,25,20);
    plot(f_toplot,log10(temp),'Color',cols(iF,:),'LineWidth',2);hold on;
    plot(f_toplot,log10(temp2),'Color',cols(iF,:),'LineWidth',2,'LineStyle',':');hold on;
    set(gca,'YTick',log10(10.^[-5:1:-1]));
    set(gca,'YTickLabel',10.^[-5:1:1]);
    plot4paper('Frequency (Hz)','PSD')
end
print([figdir,'Fig6C_',int2str(imethod)','_example'],'-dpng');


% also compute the group averages:
clear acc_realonly_f acc_all_f MI_realonly_f MI_all_f
for iSj=1:15
    fprintf(['Sj=',int2str(iSj),'\n'])
    % load real part:
    indcomp = 1;
    for icond1 = 1:ncond
        for icond2=(icond1+1):ncond
            acc_realonly_f(:,:,indcomp,iSj) = pwelch(acc_realonly(:,t_to_run,indcomp,iSj)'-0.5,25,20);
            acc_all_f(:,:,indcomp,iSj) = pwelch(acc_all(:,t_to_run,indcomp,iSj)'-0.5,25,20);
            indcomp = indcomp+1;
        end
    end
end

temp_acc_all = permute(mean(log10(acc_all_f),3),[1,2,4,3]);
temp_acc_real = permute(mean(log10(acc_realonly_f),3),[1,2,4,3]);
f_toplot = linspace(0,50,size(acc_all_f,1));
figure('Position', [7 656 1911 442]);
for iF=2:nF
    subplot(1,nF-1,iF-1);
    plot(f_toplot,(mean((temp_acc_all(:,iF,:)),3)),'LineWidth',2,'Color',cols(iF,:));hold on;
    plot(f_toplot,(mean((temp_acc_real(:,iF,:)),3)),'LineWidth',2,'Color',cols(iF,:),'LineStyle',':');hold on;
    xlim([0,50])
    axis square;
    plot4paper('Freq (Hz)','Accuracy PSD')
    set(gca,'YTick',log10(10.^(-3:-1)));
    set(gca,'YTickLabels',10.^(-2:0));
    ylim([-3.4,-1.4])
    YL = ylim;
    
    corrp = osl_clustertf(permute(temp_acc_all(:,iF,:) - temp_acc_real(:,iF,:),[3,2,1]));
    %ylim([0,YL(2)]);
    title(labelsplot{iF});
    hold on;
    p_toplot = [corrp>0.975]*1;
    plot(f_toplot(find(p_toplot)),(YL(1)+0.035*diff(YL))*ones(sum(p_toplot),1),'LineWidth',4,'Color',[0.4660, 0.6740, 0.1880]);
    
    corrp = osl_clustertf(permute(temp_acc_real(:,iF,:) - temp_acc_all(:,iF,:),[3,2,1]));
    title(labelsplot{iF});
    hold on;
    p_toplot = [corrp>0.975]*1;
    plot(f_toplot(find(p_toplot)),(YL(1)+0.035*diff(YL))*ones(sum(p_toplot),1),'LineWidth',4,'Color',[0.3010, 0.7450, 0.9330]);
    
    title(labelsplot{iF});
    if iF==3 
        plot(2*[10,10], YL,'r');
        
    elseif iF==4 
        plot(2*[20,20],YL,'r');
    elseif iF==6
        plot(2*[10,10], YL,'r--');
    elseif iF==5
        plot(2*[20,20],YL,'r--');
    end
    
end
clear h;
h(1) = plot(nan,nan,'LineWidth',2,'Color',cols(1,:));
h(2) = plot(nan,nan,'LineWidth',2,'Color',cols(1,:),'LineStyle',':');
l = legend(h,{'Complex Spectrum','Real spectrum'})
set(l,'Position', [0.4680 0.0282 0.1164 0.1233])
print([figdir,'Fig6C_',int2str(imethod)','Accvsfreq_long'],'-dpng');
h2(1) = plot(nan,nan,'r');
h2(2) = plot(nan,nan,'r--');
l2 = legend(h2,{'Harmonic frequency','Aliased harmonic frequency'})
set(l2,'Position', [0.4680 0.0282 0.1164 0.1233])
print([figdir,'Fig6C_',int2str(imethod)','Accvsfreq_long2'],'-dpng');

%% FIGURE 7: Plot results of aggregate decoding
nF = 7;
freq_bands = [nan,0:10:50];
ntrees = 100;
for iSj=1:15
load([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\AggregateDec_N',int2str(ntrees),'.mat'],'acc_orig','acc_tree');

temp_acc_all(:,1:nF,iSj) = mean(acc_orig,3);
temp_acc_all(:,nF+1,iSj) = mean(acc_tree,2);

end


figure('Position',[357 494 883 604]);
cols = parula(nF);
cols = [0,0,0;cols(1:nF-1,:);0*[1,1,1]];
clear h;
temp_acc_all(setdiff(1:111,t_to_run),:,:) = NaN;
for iF=1:nF
    if iF>1
        shadedErrorBar(t_points,mean(temp_acc_all(:,iF,:),3),std(temp_acc_all(:,iF,:),[],3)./sqrt(15),{'Color',cols(iF,:),'LineWidth',2},0.5);
        hold on;
        h(iF) = plot(nan(2,1),nan(2,1),'Color',cols(iF,:),'LineWidth',2);
    else
        shadedErrorBar(t_points,mean(temp_acc_all(:,iF,:),3),std(temp_acc_all(:,iF,:),[],3)./sqrt(15),{'Color',cols(iF,:),'LineWidth',2,'LineStyle',':'},0.5);
        hold on;
        h(iF) = plot(nan(2,1),nan(2,1),'Color',cols(iF,:),'LineWidth',2,'LineStyle',':');
    end
    
end
shadedErrorBar(t_points,mean(temp_acc_all(:,nF+1,:),3),std(temp_acc_all(:,nF+1,:),[],3)./sqrt(15),{'Color',cols(nF+1,:),'LineWidth',2},0.5);
h(nF+1) = plot(nan(2,1),nan(2,1),'Color',cols(nF+1,:),'LineWidth',2);
ylim([0.47,1.05*max(squash(mean(temp_acc_all,3)))]);plot4paper('Time(sec)','Accuracy');
labelsplot{1} = 'Wideband';
for i=2:7
    labelsplot{i} = [int2str(freq_bands(i)),' Hz'];
end
labelsplot{nF+1} = 'Aggregate';
legend(h,labelsplot)
xlim([0,0.5])
title('Mean +/- ste over subjects');
print([figdir,'Fig7A_aggregatedecoding_n',int2str(ntrees),'_allsj'],'-dpng');

% also plot difference:
AggMinWB = temp_acc_all(:,8,:)-temp_acc_all(:,1,:);
figure('Position',[357 494 694 604]);
clear h;

bestf = zeros(length(t_points),1);
for t=t_to_run
    [~,bestf(t)] = max(mean(temp_acc_all(t,2:7,:),3));
    AggMinMaxf(t,1,:) = temp_acc_all(t,8,:)-max(temp_acc_all(t,bestf(t)+1,:),[],2);
    [~,pvals2(t)] = ttest(squeeze(AggMinMaxf(t,1,:)));
    [~,pvals1(t)] = ttest(squeeze(AggMinWB(t,1,:)));
end
thresh = 1.8;
[corrp] = osl_clustertf(permute(AggMinWB, [3,2,1]),thresh,1000);
[corrpmin] = osl_clustertf(permute(-AggMinWB, [3,2,1]),thresh,1000);
[corrp2] = osl_clustertf(permute(AggMinMaxf, [3,2,1]),thresh,1000);
[corrp2min] = osl_clustertf(permute(-AggMinMaxf, [3,2,1]),thresh,1000);
subplot(2,1,1);
shadedErrorBar(t_points,mean(AggMinWB(:,1,:),3),std(AggMinWB(:,1,:),[],3)./sqrt(15),{'Color',cols(1,:),'LineWidth',2},0.5);
hold on;
p_toplot = [corrp>0.975]*1;
plot(t_points(find(p_toplot)),-0.015*ones(sum(p_toplot),1),'LineWidth',4);
p_toplot = [corrpmin>0.975]*1;
plot(t_points(find(p_toplot)),-0.015*ones(sum(p_toplot),1),'LineWidth',4);
plot4paper('Time(sec)','Change in accuracy');
title('Aggregate > Wideband');
plot(t_points(t_to_run),zeros(length(t_to_run)),'k--');
subplot(2,1,2);
shadedErrorBar(t_points,mean(AggMinMaxf(:,1,:),3),std(AggMinMaxf(:,1,:),[],3)./sqrt(15),{'Color',cols(1,:),'LineWidth',2},0.5);
hold on;
p_toplot = [corrp2>0.975]*1;
plot(t_points(find(p_toplot)),-0.015*ones(sum(p_toplot),1),'LineWidth',4);
p_toplot = [corrp2min>0.975]*1;
plot(t_points(find(p_toplot)),-0.015*ones(sum(p_toplot),1),'LineWidth',4);
plot4paper('Time(sec)','Change in accuracy');
plot(t_points(t_to_run),zeros(length(t_to_run)),'k--');
xlim([0,0.5])
title('Aggregate > Maximum narrowband');
print([figdir,'Fig7B_aggregatedecoding_n',int2str(ntrees),'_allsj',subgroupstring,binstring,'_comparison'],'-dpng');
