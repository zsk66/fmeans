function [Coreset,weight,idx] = Coreset_Construction(X,k,dist,alpha,Init_label,dist_cluster_k,m)
[n, d] = size(X);
dist_sum = sum(dist);
c_phi = dist_sum/n;
s = zeros(n,1);
p = zeros(n,1);
weight_all= zeros(n,1);
dist_sum_Bi = zeros(n,1);
Num_Bi = zeros(n,1);
Num_cluster = zeros(k,1);
for j = 1:k
    Num_cluster(j) = sum(Init_label==j);
end
for i = 1:n
    dist_sum_Bi(i) = dist_cluster_k(Init_label(i));
    Num_Bi(i) = Num_cluster(Init_label(i));
    s(i)=alpha*dist(i)/c_phi + 2*alpha*dist_sum_Bi(i)/(Num_Bi(i)*c_phi) + 4*n/Num_Bi(i);
end
sum_s = sum(s);
for ii = 1:n
    p(ii)=s(ii)/sum_s;
    weight_all(ii) =1/(p(ii)*m);
end
[Coreset,idx]= datasample(X,m,'Replace',false,'Weights',p);
weight = weight_all(idx);
end