function MI = computeMIfunc(alpha)
% Computes the mutual information, given the product 
% alpha = 2*mu^T*Sigma*mu
%
% MI = log(2) - integral[N(u|alpha,2alpha) log(1+exp(-u))]

m = alpha;
s = sqrt(2*alpha);

u = linspace(m-4*s,m+4*s,1000);
res = u(2)-u(1);

P = normpdf(u,m,s);
res = 1./sum(P);

MI = log(2) - res*sum(P.*log(1+exp(-u)));

end