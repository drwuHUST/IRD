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