%% Decoding analysis of MEG Data
    
% This script runs the decoding analysis of Higgins et al 2022 on the data
% publicaly available at the following URL. This data should be downloaded
% to the 'rawdatadir' specified below.

%spectralmethod = 'stft'; % delete this later

% Set directories:
rawdatadirdir = 'F:\My Data\Cichy2014\';
workingdir = 'C:\Users\chiggins\Documents\Cichy2020Analysis\';

Spectdatafolder = '\STFTdata\'; % this is where we save the STFT output


% note the following function definition clashes with matlab's default -
% make sure it is removed:
rmpath([osldir, '\spm12\external\fieldtrip\external\signal']);

%setup stft params:
downsamplefactor = 10;
win = hamming(10,'periodic');
overlaplength = 9;
Fs = 100; % sampling frequency
[~,freq_bands] = stft(randn(111,100),Fs,'Window',win,'OverlapLength',overlaplength);
freq_bands = [nan;freq_bands(find(freq_bands==0):end)]; % first entry is the broadband data

% data parameters:
nTr = 30;
nCh = 306;
ttrial = 111;
ncond = 118;
nF = length(freq_bands);

%% Prepare the data:
% loop over subjects:
for iSj= 1:15
    
    for itype = 1:ncond    
        fprintf(['\nProcessing condition ',int2str(itype),' of ',int2str(ncond)]);
        data = zeros(nCh*2,ttrial*nTr,nF);
        % concatenate all data (with and without stft) for each condition
        % and save:
        for itrial = 1:30
            load([rawdatadirdir,'subj',sprintf('%02d',iSj),'\sess01\cond',sprintf('%04d',itype),'\trial',sprintf('%03d',itrial),'.mat'])
            tempdata = resample(F',1,downsamplefactor);
            %fit stft:
            [tempdatatf,f] = stft(tempdata,Fs,'Window',win,'OverlapLength',overlaplength);
            tempdatatf = tempdatatf(find(f==0):end,:,:);
            tempdatatf = cat(2,zeros(nF-1,5,nCh),tempdatatf);
            tempdatatf = cat(2,tempdatatf,zeros(nF-1,4,nCh));
            datatocat = permute(cat(3,real(tempdatatf),imag(tempdatatf)),[3,2,1]);
            timeseriestocat = [tempdata';zeros(size(tempdata'))]; %zeros are to align to imaginary part of TF transform
            data(:,(itrial-1)*ttrial + [1:ttrial],:) = cat(3,timeseriestocat,datatocat);
            fnamesuffix = ['_win',int2str(length(win))];
        end
        % save to file:
        mkdir([workingdir,'subj',sprintf('%02d',iSj),Spectdatafolder]);
        save([workingdir,'subj',sprintf('%02d',iSj),Spectdatafolder,'cond',sprintf('%04d',itype),fnamesuffix,'.mat'],'data');
    end
    
end

%% Run the analysis:
t_points = -0.1:0.01:1; % epoch timings
t_to_run = find(t_points>=0 & t_points<=0.5); % the points we will decode
for iSj=1:15
    % create a directory for this subject's results:
    mkdir(['C:\Users\chiggins\Documents\Cichy2020Analysis\subj',sprintf('%02d',iSj),'\DecRes'])
    
    % set decoding options:
    opts = [];
    opts.classifier = 'SVM';
    opts.NCV = 3; % number cross validation folds
    
    DM = [ones(30*nF,1);zeros(30*nF,1)]; % design matrix
    F = repmat(nF,1,60);
    ncomparisons = (ncond.^2-ncond)/2;
    indcomp = 1;
    
    % we now iterate over each pair of conditions:
    for icond1 = 1:ncond
        for icond2 = (icond1+1):ncond
            % first we run complex spectrum decoding, frequency by
            % frequency, feeding both real and imaginary parts of the stft
            % output into the classifer:
            acc = zeros(nF,ttrial,ncomparisons);
            preds = zeros(nF,ttrial,60);
            % load data and reshape:
            load([workingdir,'subj',sprintf('%02d',iSj),Spectdatafolder,'cond',sprintf('%04d',icond1),fnamesuffix,'.mat'],'data');;
            temp2 = load([workingdir,'subj',sprintf('%02d',iSj),Spectdatafolder,'cond',sprintf('%04d',icond2),fnamesuffix,'.mat'],'data');
            data = cat(2,data,temp2.data);
            data = reshape(data,[nCh*2,ttrial,60,nF]);
            data =  permute(data,[4,3,1,2]);
            data = reshape(data,[nF*60,nCh*2,ttrial]);
            
            acc_across = zeros(nF,ttrial,2);
            for t = t_to_run
                fprintf(['\nDecoding condition ',int2str(indcomp),' of ',int2str(ncomparisons),': t=',int2str(t)]);
                % treat frequencies as if timepoints in normal temp gen
                % pipeline:
                [acc(:,t,indcomp),~,~,~,~,Y_preds] = standard_classification(data(:,:,t),DM,F,opts);
                
                % save directional info:
                truepreds = (Y_preds>0)==reshape(DM,nF,60);
                acc_across(:,t) = mean(truepreds,2);
                preds(:,t,:) = Y_preds;
            end
            
            save([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\acc_preds',sprintf('%04d',indcomp),'.mat'],'preds','icond1','icond2');
            
            % now also run narrowband signal decoding, where only the real
            % output of the STFT is fed to the classifiers:
            acc = zeros(nF,ttrial,ncomparisons);
            preds = zeros(nF,ttrial,60);
            % only take real part:
            data = data(:,1:nCh,:);
            acc_across = zeros(nF,ttrial);
            for t = t_to_run
                fprintf(['\nDecoding condition ',int2str(indcomp),' of ',int2str(ncomparisons),': t=',int2str(t)]);
                % treat frequencies as if timepoints in normal temp gen
                % pipeline:
                [acc(:,t,indcomp),~,~,~,~,Y_preds] = standard_classification(data(:,:,t),DM,F,opts);
                
                % save accuracy info:
                truepreds = (Y_preds>0)==reshape(DM,nF,60);
                acc_across(:,t) = mean(truepreds,2);
                preds(:,t,:) = Y_preds;
            end
            save([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\acc_preds',sprintf('%04d',indcomp),'_realonly','.mat'],'preds','icond1','icond2');
            
            % increment counter:
            indcomp = indcomp+1;
        end
    end
    
end

%% Final part of analysis: aggregate decoding accuracy

% We now train a ensemble classifier to aggregate the data over frequency
% bands:

% this requires training on large data inputs, so we first set the data
% into a suitable format:
for iSj=1:15
    % load all data and save in datastore:
    indcomp = 1;
    csvfile = [workingdir,'subj',sprintf('%02d',iSj),'\preds',binstring,'.csv'];
    
    clear labels
    labels = {'Cond'};
    for i1=1:nF
        for t=t_to_run
            labels{(t-min(t_to_run))*nF + i1 + 1} = ['F',int2str(i1),'T',int2str(t)];
        end
    end
    
    if ~isfile(csvfile) 
        for icond1 = 1:ncond
            for icond2=(icond1+1):ncond
                load([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\acc_preds',sprintf('%04d',indcomp),decsavestring,'.mat'],'preds');
                %setup data format:
                temp = zeros(60,(length(t_to_run)*nF)+1);
                for t=t_to_run
                    temp([1:60],(t-min(t_to_run))*nF + [1:nF] + 1) = permute(preds(:,t,:),[3,1,2]);
                end
                temp(:,1) = repmat(indcomp,60,1);
                if indcomp==1 % initialise:
                    T = array2table(temp);
                    T.Properties.VariableNames(1:length(labels)) = labels;
                    writetable(T,csvfile);
                else
                    dlmwrite(csvfile,temp,'delimiter',',','-append');
                end
                indcomp = indcomp+1;
            end
        end
     end

end

% now run the analysis, subject by subject:
for iSj=1:15
    % and run analysis on datascore:
    nFold = 10; % number of bootstrap samples with replacement to take
    nTest = 20; % number of conditions to remove on each fold
    ntrees = 100;
    DM = [ones(30,1);zeros(30,1)];
    acc_orig = zeros(length(t_points),nF,nFold);
    acc_tree = zeros(length(t_points),nFold);
    for t=t_to_run
        labels = cell(1,nF);
        for iF = 1:nF
            labels{iF} = ['F',int2str(iF),'T',int2str(t)];
        end

        if testsubgroup
            dsfull = datastore(csvfile);
            dsfull.SelectedVariableNames = labels;
            Tfull = tall(dsfull);
            labels = labels(2:3); % 2 and 3 correspond to 0 and 10Hz bands
            subgroupstring = '_0and10Hzonly';
        else
            subgroupstring = '';
        end


        ds = datastore(csvfile);
        ds.SelectedVariableNames = labels;
        T = tall(ds);
        for iFold = 1:nFold
            fprintf(['\nDecoding for t=',num2str(t_points(t)),', fold ',int2str(iFold)]);
            % bootstrap sample different conditions:
            condlist = randperm(ncond);
            testset = sort(condlist(1:nTest));
            trainset = sort(condlist((nTest+1): end));
            % create boolean indices
            testselect = false(ncomparisons*60,1);
            trainselect = false(ncomparisons*60,1);
            indcomp=1;
            for icond1=1:ncond % iterate over all pairs of conditions creating flags for training and test data:
                for icond2 =icond1+1:ncond
                    if ismember(icond1,trainset) && ismember(icond2,trainset)
                        trainselect((indcomp-1)*60 + [1:60]) = true;
                    elseif ismember(icond1,testset) && ismember(icond2,testset)
                        testselect((indcomp-1)*60 + [1:60]) = true;
                    end
                    indcomp = indcomp+1;
                end
            end

            % and fit random forrest classifier:
            n_train = ((ncond-nTest).^2 - (ncond-nTest))/2;
            n_test = ((nTest).^2 - (nTest))/2;
            fulltruescores = tall(repmat(DM,ncomparisons,1));
            B = TreeBagger(ntrees,T(trainselect,:),fulltruescores(trainselect,:));
            preds = gather(B.predict(T(testselect,:)));
            preds_int = zeros(length(preds),1);
            for i=1:length(preds)
                if strcmp(preds{i},'1')
                   preds_int(i) = 1;
                else
                    preds_int(i) = 0;
                end
            end
            DM_test = repmat(DM,n_test,1);
            testdata = table2array(gather(T(testselect,:)));
            acc_orig(t,:,iFold) = mean(~xor(repmat(logical(DM_test),1,nF),testdata>0));
            acc_tree(t,iFold) = mean(~xor(preds_int,logical(DM_test)));
        end
    end
    save([workingdir,'subj',sprintf('%02d',iSj),'\DecRes\AggregateDec_N',int2str(ntrees),'.mat'],'acc_orig','acc_tree');
end

