%======================================================================
% Compute Spectral Radius for One-hidden-layer Networks
%======================================================================
function [srad,A] = Compute_srad(pw,pu,rhow,rhou,rhox,K,alpha,ROWNORMW,ROWNORMU)
% p_w = pnorm of output units
% p_u = pnorm of hidden unit
% rho_w = radius of W on the p_w ball
% rho_u = radius of U on the p_u ball
% X = data matrix of size N times dim (N number of samples)
% Y label vector of dimension N (label encoding from 1 to K)
% alpha = exponents in the classifier
% ROWNOMW - if 1 bound for row-wise normalization of w, 0 if not
% ROWNOMU - if 1 bound for row-wise normalization of u, 0 if not

n1 = length(alpha);
puprime=pu/(pu-1);
pwprime=pw/(pw-1);

% this calculation of psi1 is different from paper
% ixu = find(alpha>=pu/pwprime);
% ixv = find(alpha<pu/pwprime);
% if(length(ixv)>0)
%     part1=max( (rhou*rhox).^(pwprime*alpha(ixv))).*length(ixv).^(1-pwprime*min(alpha(ixv))/pu);
% else
%     part1=0;
% end
% if(length(ixu)>0)
%     part2=max( (rhou*rhox).^(pwprime*alpha(ixu)));
% else
%     part2=0;
% end
% maxoutput1 = rhow*(part1+part2).^(1/pwprime);

maxoutput1 = rhow*norm( (rhou*rhox).^alpha,pwprime);
maxoutput2 = rhow*norm( (rhou*rhox).^alpha,pwprime);
secondbd = rhow*norm(alpha.*(rhou*rhox).^alpha,pwprime);

if(ROWNORMW==1)
    if(ROWNORMU==0)
        A = zeros(K+1,K+1);
        A(1:K,1:K) = 4*(pwprime-1)*maxoutput1;    % A(w,w)
        A(1:K,K+1) = 2*(pwprime-1)*(max(alpha) + 2*secondbd); % A(w,u)
        A(K+1,1:K) = 2*(puprime-1)*(2*maxoutput1 + 1);           % A(u,w)
        A(K+1,K+1) = 2*(puprime-1)*(max(abs(alpha-1)) + 2*secondbd); % A(u,u)
    else
        A = zeros(K+n1,K+n1);
        A(1:K,1:K)          = 4*(pwprime-1)*maxoutput2;
        A(1:K,(K+1):(K+n1)) = 2*(pwprime-1)*(max(alpha) + 2*secondbd);
        A((K+1):(K+n1),1:K) = 2*(puprime-1)*(2*maxoutput2 + 1);
        A((K+1):(K+n1),(K+1):(K+n1)) = 2*(puprime-1)*(max(abs(alpha-1)) + 2*secondbd);
    end
end
if(ROWNORMW==0)
    if(ROWNORMU==1)
        A=zeros(1+n1,1+n1);
        A(1,1) = 4*(pwprime-1)*maxoutput2;    % A(w,w)
        A(1,2:(n1+1)) = 2*(pwprime-1)*(max(alpha) + 2*secondbd); % A(w,u)
        A(2:(n1+1),1) = 2*(puprime-1)*(2*maxoutput2 + 1);           % A(u,w)
        A(n1+1,n1+1) = 2*(puprime-1)*(max(abs(alpha-1)) + 2*secondbd); % A(u,u)
    else
        A=zeros(2,2);
        A(1,1) = 4*(pwprime-1)*maxoutput1;    % A(w,w)
        A(1,2) = 2*(pwprime-1)*(max(alpha) + 2*secondbd); % A(w,u)
        A(2,1) = 2*(puprime-1)*(2*maxoutput1 + 1);           % A(u,w)
        A(2,2) = 2*(puprime-1)*(max(abs(alpha-1)) + 2*secondbd); % A(u,u)
    end
end
if sum(isnan(A(:)))>0 || sum(isinf(A(:)))>0
    disp('matrix A contains unnumerical values');
    min(alpha), max(alpha)
    pause
end

srad=max(abs(eig(A)));


