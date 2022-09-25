import cv2
import matplotlib.pyplot as plt

def splitImage(imagePath="images/fig1.jpg",blocksH=2,blocksW=2,w=256,h=256):
    im =  cv2.imread(imagePath)
    im = cv2.resize(im,(w,h))

    imgheight=im.shape[0]
    imgwidth=im.shape[1]

    M = imgheight//blocksH
    N = imgwidth//blocksW
    images=[]
    for y in range(0,imgheight,M):
        for x in range(0, imgwidth, N):
            images.append(im[y:y+M,x:x+N])

    return images

path_imgNo="/home/poste6/Bureau/TP3_IM/brain_tumor_dataset/no/5 no.jpg"
path_imgYes="/home/poste6/Bureau/TP3_IM/brain_tumor_dataset/yes/Y20.jpg"

blocksH=1
blocksW=2
l_images= splitImage(imagePath=path_imgYes,blocksH=blocksH,blocksW=blocksW)
fig = plt.figure()

for i in range(blocksH*blocksW):
    ax = fig.add_subplot(blocksH, blocksW, i+1);
    plt.imshow(l_images[i]);
    ax.set_axis_off();

plt.tight_layout()
plt.show()
