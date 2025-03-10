function f=RDCT(F)

[M, N] = size(F);

f = zeros(M, N);
    
x = 0:M-1;
y = 0:N-1;
    
[X, Y] = meshgrid(x, y);
CX = sqrt(2/M) * (X > 0) + 1/sqrt(M);
CY = sqrt(2/N) * (Y > 0) + 1/sqrt(N);
    
for u = 0:M-1
    for v = 0:N-1
            f(u+1, v+1) = sum(sum(F .* cos(pi/M * (u + 0.5) .* X) .* cos(pi/N * (v + 0.5) .* Y))) * CX(u+1, v+1) * CY(u+1, v+1);
    end
end
end
