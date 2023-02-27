function [C, I, iter,Loss,final_d2,minDist,final_weighted_loss,error,running_time] = fmeans(data, K, maxIter,...
    group_num,coreset_flag,Init_label,C,sum_d,All_Dist,theta,m,weight_flag)
% number of vectors in X
[n, dim] = size(data);
All_Dist = min(All_Dist,[],2);
group_idx = zeros(n, 1);
w_dist = zeros(n, 1);

%% Compute Weights
if strcmp(weight_flag,'ed')
for j = 1:K
    temp_idx = find(Init_label==j);
    if isempty(temp_idx)
        continue
    end
    Dist_in_cluster_j=All_Dist(temp_idx);
    Dist_max = max(Dist_in_cluster_j);
    [Num_cluster_j,~] = size(temp_idx);
    for i = 1:Num_cluster_j
        data_id = temp_idx(i);
        if Dist_max==0
        group_idx(data_id)=1;
        else
        group_idx(data_id) = floor(group_num-(Dist_in_cluster_j(i)/Dist_max)*group_num)+1;
        end
        w_dist(data_id) = 1/((group_idx(data_id))^theta);
    end
end

elseif strcmp(weight_flag,'ep')
for j = 1:K
    temp_idx = find(Init_label==j);
    if isempty(temp_idx)
        continue
    end
    Dist_in_cluster_j=All_Dist(temp_idx);
    Dist_max = max(Dist_in_cluster_j);
    [Num_cluster_j,~] = size(temp_idx);
    [~,ids]=sort(Dist_in_cluster_j,'descend');
    sord_idx = temp_idx(ids);
    step = floor(Num_cluster_j/group_num);
    for i = 1:group_num
        start = (i-1)*step+1;
        if i==group_num
            stop = Num_cluster_j;
        else
            stop = i*step;
        end

        for ii = start:stop
            data_id = sord_idx(ii);
            w_dist(data_id) = 1/(i^theta);
        end
    end
end
end

weighted_dist = w_dist.*All_Dist;
Init_Loss =sum(w_dist.*All_Dist);
cluster_std = zeros(K,1);
for k=1:K
    idx = Init_label == k;
    cluster_std(k)=var(All_Dist(idx));
end
Init_d2 = sum(cluster_std);
fprintf('Initial weighted Loss=%f\n',Init_Loss)
fprintf('Initial d2=%f\n',Init_d2)

alpha =10 * max(w_dist)/min(w_dist); epsilon = 0.25; delta = 0.1;
if strcmp(coreset_flag,'true')
    mu = 1; 
    [Coreset,weight,sample_idx] = Coreset_Construction(data,K,...
        weighted_dist,alpha,Init_label,sum_d,m);
    w = w_dist(sample_idx).*weight;
    X = Coreset;
    I = Init_label(sample_idx);
    [vectors_num,~] = size(X);
elseif strcmp(coreset_flag,'uniform')
    [X,sample_idx] = datasample(data,m,'Replace',false);  
    I = Init_label(sample_idx);
    [vectors_num,~] = size(X);
    weight = (n/m)*ones(m,1);
    w = w_dist(sample_idx).*weight;
elseif strcmp(coreset_flag,'false')
    X = data;
    I = Init_label;
    [vectors_num,~] = size(X);
    w = w_dist;
end


W = repmat(w,1,dim);
WX = X.*W;

% mypar = parpool;

minDist = zeros(vectors_num, 1);
% iteration count
iter = 1;
% compute new clustering while the cumulative intracluster error in kept
% below the maximum allowed error, or the iterative process has not
% exceeded the maximum number of iterations permitted
tic
while 1
    last=I;
    % find closest point
    for i=1:vectors_num
        % find closest center to current input point
        minIdx = 1;
        minVal = norm(X(i,:) - C(minIdx,:), 2)^2;
        for j=1:K
            dist = norm(C(j,:) - X(i,:), 2)^2;
            if dist < minVal
                minIdx = j;
                minVal = dist;
            end
        end
        minDist(i) = minVal;
        % assign point to the closter center
        I(i) = minIdx;
    end


    for k=1:K
        idx = I == k;
        C(k, :) = sum(WX(idx, :));
        sum_w = sum(w(idx,:));
        C(k, :) = C(k, :) / sum_w;
    end
    
    % compute weighted RSS error
%     w_RSS_error = 0;
%     for idx=1:vectors_num
%         w_RSS_error = w_RSS_error+w(idx)*norm(X(idx, :) - C(I(idx),:), 2)^2;
%     end

    % increment iteration

    if ~any(I ~= last) 
        break;
    end
    if iter > maxIter
        iter = iter - 1;
        break;
    end
    iter = iter + 1;
end

t = toc;
% delete(mypar);

%% Compute Loss and d2
II = zeros(n,1);
for i=1:n
    % find closest center to current input point
    minIdx = 1;
    minVal = norm(data(i,:) - C(minIdx,:), 2)^2;
    for j=1:K
        dist = norm(C(j,:) - data(i,:), 2)^2;
        if dist < minVal
            minIdx = j;
            minVal = dist;
        end
    end
    minDist(i) = minVal;
    % assign point to the closter center
    II(i) = minIdx;
end
Loss = sum(minDist);
% compute d2
cluster_std = zeros(K,1);
for k=1:K
    idx = find(II ==k);
    if isempty(idx)
        continue
    end
    cluster_std(k)=var(minDist(idx));
end
final_weighted_loss = sum(w_dist.*minDist);
if strcmp(coreset_flag,'true')||strcmp(coreset_flag,'uniform')
    w_RSS_error = 0;
    for idx=1:vectors_num
        w_RSS_error = w_RSS_error+w(idx)*norm(X(idx, :) - C(I(idx),:), 2)^2;
    end
    error = abs(final_weighted_loss-w_RSS_error)/final_weighted_loss;
else
    error=0;
end
final_d2 = sum(cluster_std);
disp(['f-means took ' int2str(iter) ' steps to converge']);
fprintf('The final Loss=%f\n',final_weighted_loss)
fprintf('d2=%f\n',final_d2)
fprintf('d2 reduction=%f\n',Init_d2-final_d2)
iter = 1:iter;
running_time = t;
