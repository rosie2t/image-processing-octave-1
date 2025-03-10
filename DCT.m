function F=DCT(f)

[M, N] = size(f);

F = zeros(M, N);
    
u = 0:M-1;
v = 0:N-1;
    
[U, V] = meshgrid(u, v);
CU = sqrt(2/M) * (U > 0) + 1/sqrt(M);
CV = sqrt(2/N) * (V > 0) + 1/sqrt(N);
    
for x = 0:M-1
    for y = 0:N-1
        F(x+1, y+1) = sum(sum(f .* cos(pi/M * (x + 0.5) .* U) .* cos(pi/N * (y + 0.5) .* V))) * CU(x+1, y+1) * CV(x+1, y+1);
    end
end
end