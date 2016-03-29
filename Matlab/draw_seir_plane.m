function draw_seir_plane()
[s,x]=meshgrid(-3:.3:3,-2:.3:2);
dx=-1+0.06851662/(0.001151*s);
ds=ones(size(dx));
quiver(s,x,ds,dx);
end