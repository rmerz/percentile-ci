function [ji,ki,level] = percentile_ci(p,sigma,n)

if (n>100)
    nu = norminv((1+sigma)/2)*sqrt(p*(1-p));
    ji = floor(n*p-nu*sqrt(n));
    ki = ceil(n*p+nu*sqrt(n));
    level = sigma;
    return;
end

j_k_range = 1:n;
J = repmat(j_k_range,[n,1])';
K = repmat(j_k_range,[n,1]);
diff_Bk_Bj = binocdf(K-1,n,p)-binocdf(J-1,n,p);
% We will get too many of them
[j_all,k_all] = find(diff_Bk_Bj >= sigma);
if (length(j_all) == 0)
    ji = 0;
    ki = 0;
    level = 0;
    return;
end
% Find the minimum interval   
diff_k_j = k_all-j_all;
index_min_int = find(diff_k_j == min(diff_k_j));
j_selection = j_all(index_min_int);
k_selection = k_all(index_min_int);

% Keep the most symmetric interval
k_p = floor(p*n + (1-p));
k_pp = ceil(p*n + (1-p));
[discard,index_best_interval] = min(abs((k_p-j_selection)-(k_selection-k_pp)));

ji = j_selection(index_best_interval);
ki = k_selection(index_best_interval);
level = diff_Bk_Bj(ji,ki);

return
