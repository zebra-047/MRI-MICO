import MICO_2D.MICO as MICO
from PIL import Image
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import image

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



if __name__ == '__main__':
    iterNum=100
    N_region=3
    q=30
    ImgDir='brainweb64.tif'
    Img3= tiff.imread(ImgDir)
    Img3=np.asarray(Img3)
    A=255
    Img=Img3[:,:,0]
    Img_original=Img.copy()
    [nrow,ncol]=Img.shape
    n=nrow*ncol
    ROI=(Img>5)
    ROI=np.asarray(ROI)

    Bas=MICO.getBasisOrder3(nrow,ncol)
    N_bas=np.size(Bas,2)
    ImgG=np.zeros([nrow,ncol,N_bas])
    GGT=np.zeros([nrow,ncol,N_bas,N_bas])
    for ii in range(N_bas):
        ImgG[:,:,ii]=Img*Bas[:,:,ii]*ROI
        for jj in range(ii,N_bas):
            GGT[:,:,ii,jj]=Bas[:,:,ii]*Bas[:,:,jj]*ROI
            GGT[:,:,jj,ii]=GGT[:,:,ii,jj]

    energy_MICO=np.zeros([3,iterNum])

    b=np.ones(Img.shape)
    ini=1
    energy_MICO = np.zeros([ini, iterNum])

    # fig = plt.figure()
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False
    # ax = fig.add_subplot(1)
    # line=None
    # obsX = []
    # obsY = []

    for ini_num in range(ini):
        C=np.random.rand(3)
        C=C*A
        M=np.random.rand(nrow,ncol,3)
        a=np.sum(M,2)
        for k in range(N_region):
            M[:,:,k]=M[:,:,k]/a

        N_max=np.argmax(M,2)
        for kk in range(np.size(M,2)):
            M[:,:,kk]=(N_max==kk)

        M_old=M.copy()
        chg=10000
        energy_MICO[ini_num,0]=MICO.get_energy(Img,b,C,M,ROI,q)
        print("iter", 0, ":  ", energy_MICO[ini_num, 0], "\n")
        # obsX.append(0)
        # obsY.append(MICO.get_energy(Img,b,C,M,ROI,q))
        for n in range(1,iterNum):
            M,b,C=MICO.MICO(Img,q,ROI,M,C,b,Bas,GGT,ImgG,1,1)
            energy_MICO[ini_num,n]=MICO.get_energy(Img,b,C,M,ROI,q)

            if n%1==0:
                PC=np.zeros(Img.shape)
                for k in range(N_region):
                    PC=PC+C[k]*M[:,:,k]

                Img_im = Image.fromarray(Img)
                # plt.imshow(Img)


                BiasF_im = Image.fromarray(b*ROI)
                # plt.imshow(b*ROI)


                BiasC_im = Image.fromarray((Img/b) * ROI)
                plt.imshow(BiasC_im)


                # obsX.append(n)
                # obsY.append(energy_MICO[ini_num,n])
                # if line is None:
                #     line = ax.plot(obsX, obsY, '-g', marker='.')[0]
                # line.set_xdata(obsX)
                # line.set_ydata(obsY)
                # plt.pause(0.1)
                print("iter", n ,":  ",energy_MICO[ini_num,n],"\n")

    image.imsave('original.jpg', Img_im)
    image.imsave('bias_field.jpg', BiasF_im)
    image.imsave('bias_corrected.jpg', BiasC_im)