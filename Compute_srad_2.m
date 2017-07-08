%======================================================================
% Compute Spectral Radius for Two-hidden-layer Networks
%======================================================================
function [srad,A] = Compute_srad_2(pw,pv,pu,rhow,rhov,rhou,rhox,K,alpha,beta,param)
% param == 0 is standard normalization on U and V
% param == 1 is standard normalization on U and row normalization on V
% param == 2 is row normalization on U and standard normalization on V
% param == 3 is row normalization on U and V
% 
% 
% p_w = pnorm of output units
% p_v = pnorm of hidden unit 1
% p_u = pnorm of hidden unit 2 
% rho_w = radius of W on the p_w ball, W has dimension K x n2
% rho_v = radius of V on the p_v ball, V has dimension n2 x n1
% rho_u = radius of U on the p_u ball, U has dimension n1 x d
% rhox = maximum radius of samples on the pu-prime ball
% X = data matrix of size N times dim (N number of samples)
% K = number of clases
% alpha = exponents in the classifier for hidden layer 2
% beta = exponents in the classifier for hidden layer 1
%
% The arguments below are not used for the moment. Code is for row
% normalization on w and standard normalization on v and u

% ROWNOMW - if 1 bound for row-wise normalization of w, 0 if not
% ROWNOMU - if 1 bound for row-wise normalization of u, 0 if not



n1 = length(alpha);
n2 = length(beta);
[dumb, dumber] = size(alpha);
if dumb >= dumber
    alpha = alpha';
end
[dumb, dumber] = size(beta);
if dumb >= dumber
    beta = beta';
end

abar = min(alpha);
puprime = pu/(pu-1);
pvprime = pv/(pv-1);
pwprime = pw/(pw-1);
xi = rhou^(abar*pvprime/pu)*n1^(1-abar*pvprime/pu)*max(rhox.^(alpha*pvprime));
thetaw = rhow*norm((rhov*xi).^(beta*pwprime),pwprime);
abar = min(alpha);
if abar >pu/pvprime 
    srad = inf;
    warning('attention, min(alpha)< pu/prime should hold');
elseif param == 0
    A = zeros(K+2,K+2);
    thetav = rhow*norm((beta.^(pwprime)).*(rhov*xi).^(beta*pwprime),pwprime);
    thetau = max(alpha)*thetav;
    A(1:K,1:K)=4*(pwprime-1)*thetaw;
    A(1:K,K+1)=2*(pwprime-1)*(2*thetav+1);
    A(1:K,K+2)=2*(pwprime-1)*(2*thetau+max(alpha));
    A(K+1,1:K)=2*(pvprime-1)*(2*thetaw+1);
    A(K+1,K+1)=2*(pvprime-1)*(2*thetav-1+max(beta));
    A(K+1,K+2)=2*(pvprime-1)*(2*thetau+max(beta));
    A(K+2,1:K)=2*(puprime-1)*(2*thetaw+1);
    A(K+2,K+1)=2*(puprime-1)*(2*thetav+max(beta));
    A(K+2,K+2)=2*(puprime-1)*(2*thetau-2+max(beta)+max(alpha));
elseif param == 1
    A = zeros(K+1+n2,K+1+n2);
    thetav = rhow*beta.*((rhov*norm((rhou*rhox).^(alpha),pvprime)).^(beta));
    thetau = max(alpha)*rhow*norm((beta.^(pwprime)).*(rhov*xi).^(beta*pwprime),pwprime);
    A(1:K,1:K)=4*(pwprime-1)*thetaw;
    A(1:K,K+1:K+n2)=ones(K,1)*(2*(pwprime-1)*(2*thetav+1));
    A(1:K,K+n2+1)=2*(pwprime-1)*(2*thetau+max(alpha));
    A(K+1:K+n2,1:K)=2*(pvprime-1)*(2*thetaw+1);
    A(K+1:K+n2,K+1:K+n2)=ones(n2,1)*(2*(pvprime-1)*(2*thetav-1+max(beta)));
    A(K+1:K+n2,K+n2+1)=2*(pvprime-1)*(2*thetau-max(beta));
    A(K+n2+1,1:K)=2*(puprime-1)*(2*thetaw+1);
    A(K+n2+1,K+1:K+n2)=2*(puprime-1)*(2*thetav+max(beta));
    A(K+n2+1,K+n2+1)=2*(puprime-1)*(2*thetau-2+max(beta)+max(alpha));
elseif param == 2
    A = zeros(K+n1+1,K+n1+1);
    thetav = rhow*norm((beta.^(pwprime)).*(rhov*xi).^(beta*pwprime),pwprime);
    thetau = alpha*rhow*norm((rhov*xi).^(beta*pwprime),pwprime);
    A(1:K,1:K)=4*(pwprime-1)*thetaw;
    A(1:K,K+1)=2*(pwprime-1)*(2*thetav+1);
    A(1:K,K+2:K+1+n1)=ones(K,1)*(2*(pwprime-1)*(2*thetau+max(alpha)));
    A(K+1,1:K)=2*(pvprime-1)*(2*thetaw+1);
    A(K+1,K+1)=2*(pvprime-1)*(2*thetav-1+max(beta));
    A(K+1,K+2:K+1+n1)=2*(pvprime-1)*(2*thetau+max(beta));
    A(K+2:K+1+n1,1:K)=2*(puprime-1)*(2*thetaw+1);
    A(K+2:K+1+n1,K+1)=2*(puprime-1)*(2*thetav+max(beta));
    A(K+2:K+1+n1,K+2:K+1+n1)=ones(n1,1)*(2*(puprime-1)*(2*thetau-2+max(beta)+max(alpha)));
elseif param == 3
    A = zeros(K+n1+n2,K+n1+n2);
    thetav = rhow*beta.*((rhov*norm((rhou*rhox).^(alpha),pvprime)).^(beta));
    thetau = alpha*rhow*norm((rhov*xi).^(beta*pwprime),pwprime);
    A(1:K,1:K)=4*(pwprime-1)*thetaw;
    A(1:K,K+1:K+n2)=ones(K,1)*2*(pwprime-1)*(2*thetav+1);
    A(1:K,K+n2+1:K+n1+n2)=ones(K,1)*2*(pwprime-1)*(2*thetau+max(alpha));
    A(K+1:K+n2,1:K)=2*(pvprime-1)*(2*thetaw+1);
    A(K+1:K+n2,K+1:K+n2)=ones(n2,1)*(2*(pvprime-1)*(2*thetav-1+max(beta)));
    A(K+1:K+n2,K+n2+1:K+n1+n2)=ones(n2,1)*(2*(pvprime-1)*(2*thetau+max(beta)));
    A(K+n2+1:K+n1+n2,1:K)=2*(puprime-1)*(2*thetaw+1);
    A(K+n2+1:K+n1+n2,K+1:K+n2)=ones(n1,1)*(2*(puprime-1)*(2*thetav+max(beta)));
    A(K+n2+1:K+n1+n2,K+n2+1:K+n1+n2)=ones(n1,1)*(2*(puprime-1)*(2*thetau-2+max(beta)+max(alpha)));
end

if sum(isnan(A(:)))>0 || sum(isinf(A(:)))>0
    disp('matrix A contains unnumerical values');
    min(alpha), max(alpha), min(beta), max(beta)
    pause;
end

srad=max(abs(eig(A)));
    