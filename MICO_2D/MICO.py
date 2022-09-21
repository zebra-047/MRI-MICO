import numpy as np
import math

def MICO(Img,q,W,M,C,b,Bas,GGT,ImgG,Iter,IterCM):
    N_class = np.size(M, 2)
    for i in range(Iter):
        C=updateC(Img,W,b,M)
        [aa,bb]=Img.shape
        D = np.zeros([aa,bb,N_class])
        for k in range(IterCM):
            # e=np.zeros(M.shape)
            for kk in range(N_class):
                D[:,:,kk]=(Img-C[kk]*b)**2
            M=updateM(D,q)
    b=updateB(Img,q,C,M,Bas,GGT,ImgG)
    return M,b,C


def updateC(Img,W,b,M):
    N_Class=np.size(M,2)
    C_new=np.zeros(N_Class)
    for nn in range(N_Class):
        N=b*Img*M[:,:,nn]
        D=(b**2)*M[:,:,nn]
        sN=np.sum(N*W)
        sD=np.sum(D*W)
        C_new[nn]=sN/(sD+(sD==0))
    return C_new

def updateM(e,q):
    N_class=np.size(e,2)
    M=np.zeros(e.shape)
    if q>1:
        epsilon=1e-12
        # 避免除零
        e=e+epsilon
        p=1/(q-1)
        f=1/(e**p)
        f_sum=np.sum(f,axis=2)
        for kk in range(N_class):
            M[:,:,kk]=f[:,:,kk]/f_sum
    elif q==1:
        N_min=e.argmin(2)
        for kk in range(N_class):
            M[:,:,kk]=(N_min==kk)
    else:
        assert False,'MICO: wrong fuzzifizer'

    return M

def updateB(Img,q,C,M,Bas,GGT,ImgG):
    PC2=np.zeros(Img.shape)
    PC=np.zeros(Img.shape)
    N_class=np.size(M,2)
    for kk in range(N_class):
        PC2=PC2+(C[kk]**2)*(M[:,:,kk]**q)
        PC=PC+C[kk]*(M[:,:,kk]**q)

    N_bas=np.size(Bas,2)
    V=np.zeros(N_bas)
    A=np.zeros([N_bas,N_bas])
    for ii in range(N_bas):
        ImgG_PC=ImgG[:,:,ii]*PC
        V[ii]=np.sum(ImgG_PC)
        for jj in range(ii,N_bas):
            B=GGT[:,:,ii,jj]
            A[ii,jj]=np.sum(B)
            A[jj,ii]=A[ii,jj]
    w=np.dot(np.linalg.inv(A),V)
    b=np.zeros(Img.shape)
    for kk in range(N_bas):
        b=b+w[kk]*Bas[:,:,kk]
    return b

def get_energy(Img,b,C,M,ROI,q):
    N=np.size(M,2)
    energy=0
    ones=np.ones(Img.shape)
    for k in range(N):
        C_k=C[k]*ones
        energy=energy+np.sum(np.sum(((Img*ROI-b*C_k*ROI)**2)*(M[:,:,k]**q)))
    return energy

def getBasisOrder3(Height,Wide):
    x=np.zeros([Height,Wide])
    for i in range(Height):
        for j in range(Wide):
            x[i,j]=-1+j*2/Wide
    y = np.zeros([Height,Wide])
    for i in range(Wide):
        for j in range(Height):
            y[j,i]=-1+j*2/(Height)

    bais=np.zeros([Height,Wide,10])
    bais[:, :, 0] = np.ones([Height,Wide])
    bais[:, :, 1] = x
    bais[:, :, 2] = ((x**2)*3-1)/2
    bais[:, :, 3] = ((x**3)*5-3*x)/2
    bais[:, :, 4] = y
    bais[:, :, 5] = x*y
    bais[:, :, 6] = y*(3*x**2-1)/2
    bais[:, :, 7] = (3*y**2-1)/2
    bais[:, :, 8] = (3*y**2-1)*x/2
    bais[:, :, 9] = (5*y**3-3*y)/2

    for kk in range(10):
        A=bais[:,:,kk]**2
        r=math.sqrt(np.sum(A))
        bais[:,:,kk]=bais[:,:,kk]/r
    return bais