% laplace.m
% Author: Lorenzo Bolla <lbolla (at) gmail dot com>
% Last modified: Apr. 11, 2007
 
xmin = 0;
xmax = 1;
ymin = 0;
ymax = 1;
 
nx = 200;
ny = nx;
 
dx = (xmax - xmin) / (nx - 1);
dy = (ymax - ymin) / (ny - 1);
 
x = xmin:dx:xmax;
y = ymin:dy:ymax;
 
n_iter = 100;
eps = 1e-16;
 
u = zeros(nx,ny);
u(1,:)   = xmin^2 - y.^2;
u(end,:) = xmax^2 - y.^2;
u(:,1)   = x.'.^2 - ymin^2;
u(:,end) = x.'.^2 - ymax^2;
err = Inf;
count = 0;
while err > eps && count < n_iter
    count = count + 1;
    dx2 = dx^2;
    dy2 = dy^2;
    dnr_inv = 0.5 / (dx2 + dy2);    
    u_old = u;
    u(2:end-1, 2:end-1) = ((u(1:end-2, 2:end-1) + u(3:end, 2:end-1))*dy2 + (u(2:end-1,1:end-2) + u(2:end-1, 3:end))*dx2)*dnr_inv;
    v = (u - u_old);
    err = sqrt(v(:).'*v(:));
end

