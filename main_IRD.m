%% Implementation of the IRD algorithm in the following paper:
%%
%% 刘子昂, 蒋雪, 伍冬睿*, "基于池的无监督线性回归主动学习," 自动化学报, 2020.
%% LIU Zi-Ang, JIANG Xue, WU Dong-Rui. Unsupervised Pool-Based Active Learning for Linear Regression.
%% Acta Automatica Sinica, 2020.
%%
%% Or the English version here: https://arxiv.org/pdf/2001.05028.pdf
%%
%% Compare 4 approaches
%%
%% 1. RS (Random sampling)
%%
%% 2. GSx in the paper: D. Wu*, C-T Lin and J. Huang*, "Active Learning for Regression Using Greedy Sampling,"
%%    Information Sciences, vol. 474, pp. 90-105, 2019.
%%
%% 3. RD in the paper: D. Wu, "Pool-based sequential active learning for regression,"
%%    IEEE Trans. on Neural Networks and Learning Systems, 30(5), pp. 1348-1359, 2019.
%%
%% 4. IRD
%%
%% Dongrui Wu, drwu@hust.edu.cn


clc; clearvars; close all; rng('default');
nRepeats = 30; % number of repeats to get statistically significant results
rr = .5; % Ridge regression parameter

datasets = {'Yacht','Concrete'};
Cmax = 5;  % Maximum number of iterations for IRD; c_max in the paper
minN = 5; % mininum number of samples to select
maxN = 15; % maximum number of samples to select
nAlgs = 4; % number of algorithms to compare
RMSEsTL = cell(1,length(datasets)); CCsTL = RMSEsTL; % transductive learning
RMSEsIL = RMSEsTL; CCsIL = RMSEsIL; % inductive learning

linestyle={'k-','r-.','g--','b-'};
legends={'BL','GSx','RD','IRD'};

for s=1:length(datasets)
    s
    
    temp=load([datasets{s} '.mat']); data=temp.data;
    X0=data(:,1:end-1); Y0=data(:,end); nY0=length(Y0);
    
    RMSEsTL{s}=nan(nAlgs,nRepeats,maxN); CCsTL{s}=RMSEsTL{s};
    RMSEsIL{s}=RMSEsTL{s};  CCsIL{s}=RMSEsTL{s};
    
    for r=1:nRepeats
        fprintf('%d-',r);
        
        % random effect: 50% as pool
        ids = datasample(1:nY0,round(nY0*.5),'Replace',false);
        idsI = 1:nY0; idsI(ids) = [];  % Inductive learning
        
        % normalization
        X = X0(ids,:); Y = Y0(ids); nY = length(Y); % Transactive learning
        XI = X0(idsI,:); YI = Y0(idsI);  % Inductive learning
        [X,mu,sigma] = zscore(X);   XI = (XI-mu)./sigma;  % z-score in inductive learning
        distX=pdist2(X,X);
        
        % 1. Random selection (RS)
        idsTrain = repmat(sort(datasample(1:nY, maxN,'replace',false)),nAlgs,1);
        
        for d=minN:maxN
            
            % 2. GSx
            if d==minN
                dist = mean(distX,2);  % Initialization for GSx
                [~,idsTrain(2,1)] = min(dist);
                idsRest = 1:nY; idsRest(idsTrain(2,1)) = [];
                for n = 2:minN
                    dist = min(distX(idsRest,idsTrain(2,1:n-1)),[],2);
                    [~,idx] = max(dist);
                    idsTrain(2,n) = idsRest(idx);
                    idsRest(idx) = [];
                end
            else
                idsRest = 1:nY; idsRest(idsTrain(2,1:d-1)) = [];
                dist = min(distX(idsRest,idsTrain(2,1:d-1)),[],2);
                [~,idx] = max(dist);
                idsTrain(2,d) = idsRest(idx);
            end
            
            % 3. RD
            idsTrain(3,1:d)=RD(X,d);
            
            % 4. IRD
            idsTrain(4,1:d)=IRD(X,d,Cmax);
            
            
            %% Compute RMSEs and CCs
            for idxAlg = 1:nAlgs  % different initialization approaches
                idsRest = 1:nY; idsRest(idsTrain(idxAlg,1:d)) = [];
                mdl = fitrlinear(X(idsTrain(idxAlg,1:d),:),Y(idsTrain(idxAlg,1:d)), 'Lambda', rr, 'Learner', 'leastsquares', 'Regularization', 'ridge');
                % transductive learning
                YPredTL = Y; YPredTL(idsRest)=predict(mdl,X(idsRest,:));
                RMSEsTL{s}(idxAlg,r,d) = sqrt(mean((YPredTL-Y).^2));
                CCsTL{s}(idxAlg,r,d) = corr(YPredTL,Y);
                % inductive learning
                YPredIL = predict(mdl,XI);
                RMSEsIL{s}(idxAlg,r,d) = sqrt(mean((YPredIL-YI).^2));
                CCsIL{s}(idxAlg,r,d) = corr(YPredIL,YI);
            end
        end % end for n=minN:maxN
    end     % end for r = 1:nRepeat
    
    figure;
    subplot(1,2,1); hold on;
    for i=1:nAlgs
        plot(minN:maxN,squeeze(mean(RMSEsTL{s}(i,:,minN:maxN,1),2)),linestyle{i});
    end
    axis tight; box on;
    ylabel('RMSE'); xlabel('M');
    legend(legends,'location','northeast');
    title(datasets{s});
    
    subplot(1,2,2); hold on;
    for i=1:nAlgs
        plot(minN:maxN,squeeze(mean(CCsTL{s}(i,:,minN:maxN,1),2)),linestyle{i});
    end
    axis tight; box on;
    ylabel('CC'); xlabel('M');
    legend(legends,'location','southeast');
    title(datasets{s}); drawnow;
end



