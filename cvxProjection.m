%======================================================================
% Projection onto lp-norm sphere
%======================================================================
function z = cvxProjection(y, mask, p, rho, normType)
        if normType == 0
            mask = logical(mask);
            N = nnz(mask);
            z = zeros(size(y));
            cvx_begin quiet
                cvx_solver sedumi; 
                variable x(N, 1); 
                minimize( norm(x-y(mask)) );
                subject to
                    norm(x(:), p) <= rho;
                    x >= 0;
            cvx_end
            z(mask) = x;
        elseif normType == 1
            z = zeros(size(y));
            for i = 1:size(y, 1)
                cvx_begin quiet
                    cvx_solver sedumi; 
                    variable x(1, sum(mask(i,:)));
                    minimize( norm(x-y(i,mask(i,:))) );
                    subject to
                        norm(x, p) <= rho;
                        x >= 0;
                cvx_end
                z(i,mask(i,:)) = x;
            end
        elseif normType == 2
            z = zeros(size(y));
            for j = 1:size(y, 2)
                cvx_begin quiet
                    cvx_solver sedumi; 
                    variable x(sum(mask(:,j)), 1);
                    minimize( norm(x-y(mask(:,j),j)) );
                    subject to
                        norm(x, p) <= rho;
                        x >= 0;
                cvx_end
                z(mask(:,j),j) = x;
            end
        end
end