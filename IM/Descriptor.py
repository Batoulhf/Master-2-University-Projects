import cv2
import numpy as np
import matplotlib.pyplot as plt
#####################################################################
def SURF(img):
    surf = cv2.xfeatures2d.SURF_create()
    keypoints_surf, descriptors = surf.detectAndCompute(img, None)
    #keypoints_surf, descriptors = surf.detectAndCompute(img, None)
    print("Features : ",len(keypoints_surf))
    imgKP = cv2.drawKeypoints(img, keypoints_surf, None)
    return imgKP

#####################################################################
def SIFT(img,Filter=False):
    if Filter: img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    sift = cv2.SIFT_create()
    keypoints_sift, descriptors = sift.detectAndCompute(img, None)
    print("Features : ",len(keypoints_sift))
    imgKP = cv2.drawKeypoints(img, keypoints_sift, None)
    return imgKP
#####################################################################
def ORB(img,Filter=False):
    if Filter : img = cv2.detailEnhance(img,  sigma_s=10, sigma_r=0.15)
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints_orb, descriptors = orb.detectAndCompute(img, None)
    print("Features : ",len(keypoints_orb))


    imgKP = cv2.drawKeypoints(img, keypoints_orb, None)
    return imgKP
#cv2.imshow("Image", img);cv2.waitKey(0);cv2.destroyAllWindows()
path_imgNo="/home/dairi/Datasets/brain_tumor_dataset/no/5 no.jpg"
path_imgYes="/home/dairi/Datasets/brain_tumor_dataset/yes/Y20.jpg"
#path_imgNo="images/covid19-mini-dataset/normal/person989_virus_1667.jpeg"#NORMAL2-IM-0315-0001.jpeg"
#path_imgYes="images/covid19-mini-dataset/covid/1-s2.0-S0140673620303706-fx1_lrg.jpg"
normal=None
abnormal=None
desc='SIFT'
imgNo = cv2.detailEnhance(cv2.imread(path_imgNo),  sigma_s=10, sigma_r=0.15)
imgNo = cv2.cvtColor(imgNo, cv2.COLOR_BGR2GRAY)

imgYes = cv2.detailEnhance(cv2.imread(path_imgYes),  sigma_s=10, sigma_r=0.15)
imgYes = cv2.cvtColor(imgYes, cv2.COLOR_BGR2GRAY)


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imgNo,cmap='gray')
ax.set_title('No Anomaly')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imgYes,cmap='gray')
#imgplot.set_clim(0.0, 0.7)
ax.set_title('Anomaly')
plt.show()

if desc=='ORB':
    normal = ORB(imgNo )
    abnormal = ORB(imgYes)
if desc=='SIFT':
    normal = SIFT(imgNo)
    abnormal = SIFT(imgYes)
if desc=='SURF':
    normal = SURF(imgNo)
    abnormal = SURF(imgYes)


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(normal)
ax.set_title('No Tumor')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(abnormal)
#imgplot.set_clim(0.0, 0.7)
ax.set_title('With Tumor')
plt.show()