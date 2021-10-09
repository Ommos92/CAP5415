#Covnet

I # HxH
K # KxK
I_Out


#Convolution
for x in range(0,H):
    for y in range(0,W):
        #2D Conv
        #Not exactly how its rewritten K/2
        I_out[x,y] = I[x:x+K,y:y+K] * K
        #1D Conv Y direction
        I_out[x,y] = I[x,y:y+K] * K.transpose
        