clc
clear

%% load data
dataset_name = 'bank';
if strcmp(dataset_name,'bank')
    X = csvread('fmeans-codes/data/bank_1000_0.csv',1,0);
elseif strcmp(dataset_name,'adult')
    X = csvread('fmeans-codes/data/census_1000_0.csv',1,0);
elseif strcmp(dataset_name,'creditcard')
    X = csvread('fmeans-codes/data/creditcard_1000_0.csv',1,0);
elseif strcmp(dataset_name,'diabetes')
    X = csvread('fmeans-codes/data/diabetes_1000_0.csv',1,0);
elseif strcmp(dataset_name,'covtype')
    data = load('fmeans-codes/data/covtype');  
    X = data.X;
elseif strcmp(dataset_name,'nyctaxi')
    data = load('fmeans-codes/data/NYC-TAXI');
    X = data.X;
elseif strcmp(dataset_name,'KDD')
    data = load('fmeans-codes/data/KDD');
    X = data.X;
elseif strcmp(dataset_name,'3D')
    data = load('fmeans-codes/data/3D_spatial_network');
    X = data.X;
elseif strcmp(dataset_name,'urbanGB')
    data = load('fmeans-codes/data/urbanGB');
    X = data.X;
elseif strcmp(dataset_name,'tdrive')
    data = load('fmeans-codes/data/tdrive');
    X = data.X;
end

% X = mapminmax(X,0,1); % normalize X to [0,1]
%% hyperparameters
[n, d] = size(X);
X = X';
maxIter = 50;     % the maximum number of iterations 
group_num = 10;   % the number of groups, 'g' in our paper
c = 4;   % the number of clusters, 'k' in our paper

label_rnd = randsrc(n,1,1:c); % Random initialization for kmeans
coreset_flag='false';  % Optional: true, false, uniform
m = 10000;  % number of sampling points for coreset
theta = 0.5; % theta is the exponential term of i in our paper
weight_flag = 'ed';   % equal-distance: 'ed', equal-points: eq
%% run fmeans
% the first stage: running Lloyd's heuristic 
[Init_label,C,sum_d,All_Dist] = kmeans(X',c,'EmptyAction','error'); 

% the second stage: solving a weighted kmeans problem
[C_f, I_f, iter_f,obj_f,d2_f,minDist,final_weighted_loss,error1,running_time1] = fmeans(X', c, maxIter,...
    group_num,coreset_flag,Init_label,C,sum_d,All_Dist,theta,m,weight_flag);


