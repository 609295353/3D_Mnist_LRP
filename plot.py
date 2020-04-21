from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import matplotlib as mb
import os


# 显示切片图
def show_heatmap(img,pred,lb):
    l = len(os.listdir("./2dresult"))
    slices = []
    slice_0 = img[8, :, :]
    slice_1 = img[:, 8, :]
    slice_2 = img[:, :, 8]
    slices.append(slice_0)
    slices.append(slice_1)
    slices.append(slice_2)
    fig,axes = plt.subplots(1,len(slices))
    for i ,slice in enumerate(slices):
        ax = axes[i].imshow(slice.T,cmap="hot")
    fig.colorbar(ax, ax=axes)
    fig.suptitle("pred:" + str(pred) + "  lb:" + str(lb))
    # plt.show()
    plt.savefig("./2dresult/data_" + str(l + 1) + ".png")
    plt.close(fig)

# 寻找heatmap中
def search(heatmap,bound):
    maxx=0
    max=0
    for x in range(heatmap.shape[0]):
        n=0
        for y in range(heatmap.shape[1]):
            for z in range(heatmap.shape[2]):
                if heatmap[x,y,z]>=bound:
                    n+=1
        maxx = (x if n>max else maxx)
        max = (n if n>max else max)
    maxy = 0
    max = 0
    for y in range(heatmap.shape[1]):
        n = 0
        for x in range(heatmap.shape[0]):
            for z in range(heatmap.shape[2]):
                if heatmap[x, y, z] >= bound:
                    n += 1
        maxy = (y if n > max else maxy)
        max = (n if n > max else max)
    maxz = 0
    max = 0
    for z in range(heatmap.shape[2]):
        n = 0
        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                if heatmap[x, y, z] >= bound:
                    n += 1
        maxz = (z if n > max else maxz)
        max = (n if n > max else max)
    return maxx,maxy,maxz

# 显示切片
def show_slices(img,heatmap,pred,lb):
    bound=5
    x,y,z= 8,8,8
    l = len(os.listdir("./2dresult"))
    slices = []
    slice_0 = img[x, :, :]
    slice_1 = img[:, y, :]
    slice_2 = img[:, :, z]
    # mb.image.imsave("C:/Users/Administrator/Desktop/学习资料/脑图/slice"+str(l+1)+".png", slice_0,cmap="gray")
    # mb.image.imsave("C:/Users/Administrator/Desktop/学习资料/脑图/slice"+str(l+2)+".png", slice_1, cmap="gray")
    # mb.image.imsave("C:/Users/Administrator/Desktop/学习资料/脑图/slice"+str(l+3)+".png", slice_2, cmap="gray")
    slices.append(slice_0)
    slices.append(slice_1)
    slices.append(slice_2)
    fig,axes = plt.subplots(1,len(slices))
    for i ,slice in enumerate(slices):
        axes[i].imshow(slice.T,cmap="gray",alpha=0.6)
    slices = []
    slice_0 = heatmap[x, :, :]
    slice_1 = heatmap[:, y, :]
    slice_2 = heatmap[:, :, z]

    slices.append(slice_0)
    slices.append(slice_1)
    slices.append(slice_2)
    # for i,slice in enumerate(slices):
    #     axes[i].imshow(slice, cmap='Reds', interpolation='bilinear')
    for i, slice in enumerate(slices):
        n=0
        # data=[]
        xdata = []
        ydata = []
        zdata = []
        for x in range(slice.shape[0]):
            for y in range(slice.shape[1]):
                if slice[x,y]>=bound:
                    # data.append([x,y])
                    n+=1
                    xdata.append(x)
                    ydata.append(y)
                    zdata.append(slice[x,y])
        ax = axes[i].imshow(slice.T,cmap="hot",alpha=0.4)
        # ax = axes[i].scatter(xdata,ydata,marker=',',c=zdata,cmap=plt.cm.hot,alpha=0.2)
    fig.colorbar(ax,ax=axes)
    fig.suptitle("pred:"+str(pred)+"  lb:"+str(lb))
    # plt.show()
        # hm = HeatMap(data)alpha=(slice[x,y]/255.0)
        # hm.heatmap(base="C:/Users/Administrator/Desktop/学习资料/脑图/slice"+str(l+i+1)+".png",save_as="C:/Users/Administrator/Desktop/学习资料/脑图/heatmap"+str(l+i+1)+".png")
    plt.savefig("./2dresult/data_"+str(l+1)+".png")
    plt.close(fig)

# 获取像素点relevance大于bound的坐标x，y，z
def get_wArray(arr,bound):
    xdata = []
    ydata = []
    zdata = []
    cdata = []
    n = 0
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for z in range(arr.shape[2]):
                if arr[x,y,z]>=bound:
                    n=n+1
                    xdata.append(x)
                    ydata.append(y)
                    zdata.append(z)
                    cdata.append(arr[x,y,z])
    return xdata,ydata,zdata,cdata,n


# 显示3d heatmap
def plot3d(img,pred,lb,bound = 10):

    l = len(os.listdir("./3dresult"))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d",facecolor='white')

    xdata,ydata,zdata,cdata,n = get_wArray(img,bound)
    a = ax.scatter(xdata,ydata,zdata,marker=',',c=cdata,cmap=plt.cm.hot)
    fig.colorbar(a,ax=ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title("pred:"+str(pred)+" lb:"+str(lb))
    # plt.show()
    plt.savefig("./3dresult/data_" + str(l + 1) + ".png")
    plt.close(fig)


# 查看像素点relevance的值分布
def fenbu(heatmap):

    fenbu = np.zeros((15))
    for x in heatmap:
        for y in x:
            for z in y:
                fenbu[int(z/18)] +=1
    print(fenbu)



def test():
    img = nib.load("C:/Users/Administrator/Desktop/学习资料/数据/t1_2mm_affine_4tps/NCANDA_S00033/y1.nii.gz")
    img = img.get_fdata()
    mb.image.imsave("C:/Users/Administrator/Desktop/学习资料/nt.png",img[70,:,:],cmap='gray')

