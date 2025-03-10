function Q=midrise(Im,L)
Q=zeros(size(Im));
%L=[8 12 16 20]; %statmes kvantisth
[m,n]=size(Im);
D=256/(L+1);
x_min=min(Im(:));
x_max=max(Im(:));
D=(x_max-x_min)/L;
d_k=x_min+(0:L-1)*D;
r_k=d_k+(D/2);
for i = 1:size(Im,1)
  for j = 1:size(Im,2)
    x=Im(i,j);
    if (x<x_min)
    Q(i,j)=r_k(1);
    elseif (x>x_max)
    Q(i,j)=r_k(L);
  else
  for k= 1:length(L)
    if (d_k(k)<x && x<=d_k(k+1))
      Q(i,j)=r_k(k)=d_k(k)+D/2;
      break;
  end
end
end
end
end

