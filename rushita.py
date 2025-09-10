import g, ml, numpy

x = [1,2,3,4,5,6,7,8]
y = [0.1,0.525,0.75,0.708,0.249,0.35,0.4,0.4675]
rx = [[f] for f in x]
m, c = ml.ridreg(rx,y)

g.pxy(x,y,m[0],c)