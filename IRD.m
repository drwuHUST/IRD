function idsTrain = IRD(X, d, Cmax)

%% Implementation of the IRD algorithm in the following paper:
%%
%% 刘子昂, 蒋雪, 伍冬睿, "基于池的无监督线性回归主动学习," 自动化学报, 2020.
%% LIU Zi-Ang, JIANG Xue, WU Dong-Rui. Unsupervised Pool-Based Active Learning for Linear Regression.
%% Acta Automatica Sinica, 2020.
%%
%% Or the English version here: https://arxiv.org/pdf/2001.05028.pdf
%%
%% X: Input data matrix; each sample is a row vector
%% d: number of samples to be selected by IRD
%% Cmax: number of iterative updates; c_max in the paper
%% IdsTrain: selected sample indices
%%
%% Dongrui WU, drwu@hust.edu.cn


[N, M] = size(X);

if d <= M % Case 1
    [~,XPCA]=pca(X);
    X = XPCA(:,1:d-1);
    distX = pdist2(X,X);
    
    [idsTrain, idsP] = RD(X, d);
    R = nan(N, 1);  % representaiveness
    for n = 1:d
        R(idsP{n}) = mean(distX(idsP{n}, idsP{n}),2);
    end
    
    % Iterative improvement
    P = idsTrain;
    for c = 1:Cmax
        for n = 1: d
            [~,idx] = min(compIRD(X, idsTrain([1:n-1 n+1:end]), idsP{n}, R(idsP{n})));
            idsTrain(n) = idsP{n}(idx);
        end
        if ismember(idsTrain, P, 'rows')
            break;
        else
            P(c+1,:) = idsTrain;
        end
    end
else
    % Initialize the first M+1 points by RD
    [idsTrain0, idsP] = RD(X, M+1);
    distX = pdist2(X,X);
    R = nan(N, 1);  % representaiveness
    for n = 1:M+1
        R(idsP{n}) = mean(distX(idsP{n}, idsP{n}),2);
    end
    
    % Iteratively optimize the first M+1 points
    P = idsTrain0;
    for c = 1:Cmax
        for n = 1: M+1
            [~,idx] = min(compIRD(X, idsTrain0([1:n-1 n+1:end]), idsP{n}, R(idsP{n})));
            idsTrain0(n) = idsP{n}(idx);
        end
        if ismember(idsTrain0, P, 'rows')
            break;
        else
            P(c+1,:) = idsTrain0;
        end
    end
    
    if d==M+1 % Case 2
        idsTrain = idsTrain0; return;
    else % Case 3, select the next d-M-1 points
        idsRest = 1:N; idsRest(idsTrain0) = [];
        [idsTrain, idsP] = RD(X(idsRest, :), d-M-1);
        R = nan(N-M-1, 1);  % calculate representaiveness for step 2
        for n = 1:d-M-1
            R(idsP{n}) = mean(distX(idsRest(idsP{n}), idsRest(idsP{n})),2);
        end
        
        P = idsTrain;
        for c = 1:Cmax
            for n = 1: d-M-1
                [~, idx] = min(R(idsP{n})./...
                    min(distX(idsRest(idsP{n}), [idsTrain0, idsRest(idsTrain([1:n-1 n+1:end]))]), [], 2)); % R/D
                idsTrain(n) = idsP{n}(idx);
            end
            if ismember(idsTrain, P, 'rows')
                break;
            else
                P(c+1,:) = idsTrain;
            end
        end
        idsTrain = cat(2,idsTrain0,idsRest(idsTrain));
    end
end

function ird = compIRD(X, idsFixed, idsOpt, R)
% Compute the term on right hand side of (12)
solu = null([ones(length(idsFixed), 1), X(idsFixed,:)]);
theta = solu(:,1);
ird = R./abs([ones(length(idsOpt), 1), X(idsOpt,:)]*theta);

function [idsTrain, idsP] = RD(X, d)

%% RD in the paper:
%%
%%    D. Wu, "Pool-based sequential active learning for regression,"
%%    IEEE Trans. on Neural Networks and Learning Systems, 30(5), pp. 1348-1359, 2019.
%%
%% Dongrui Wu, drwu@hust.edu.cn

[idsClass,~,~,D] = kmeans(X,d,'MaxIter',200,'Replicates',5);
idsTrain=nan(1,d);
idsP=cell(1,d);
for n = 1:d  % find the one closest to the centroid
    idsP{n}=find(idsClass == n);
    [~,idx] = min(D(idsP{n},n));
    idsTrain(n) = idsP{n}(idx);
end

