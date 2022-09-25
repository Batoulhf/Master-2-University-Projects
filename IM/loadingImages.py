from skimage.io import imread_collection
import cv2
import os
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
    #print("Features : ",len(keypoints_sift))
    imgKP = cv2.drawKeypoints(img, keypoints_sift, None)
    return imgKP,len(keypoints_sift)
#####################################################################
def ORB(img,Filter=False):
    if Filter : img = cv2.detailEnhance(img,  sigma_s=10, sigma_r=0.15)
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints_orb, descriptors = orb.detectAndCompute(img, None)
    #print("Features : ",len(keypoints_orb))


    imgKP = cv2.drawKeypoints(img, keypoints_orb, None)
    return imgKP,len(keypoints_orb)
########################################################
def load_images_from_folder(folder,width=256, height=256):
    images = []
    names=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        if img is not None:
            images.append(img)
            names.append(filename)
    return images,names
##########################################################""
#your path

normal_dir = 'images/covid19-mini-dataset/normal/'
abnormal_dir = 'images/covid19-mini-dataset/covid/'

#normal_dir = '/home/dairi/Datasets/brain_tumor_dataset/no/'
#abnormal_dir = '/home/dairi/Datasets/brain_tumor_dataset/yes/'

#creating a collection with the available images
#col = imread_collection(normal_dir)
col_normal,files_normal = load_images_from_folder(normal_dir)
col_abnormal,files_abnormal = load_images_from_folder(abnormal_dir)
print("Normal ")
for img,filename in zip(col_normal,files_normal) :
    FimgF,nbr_features = SIFT(img,Filter=True)
    cv2.imwrite("images/features/NOR-"+str(nbr_features)+"-"+filename,FimgF)
    print(filename+" , "+str(nbr_features))


print("AbNormal ")
#for img in col_abnormal :
for img, filename in zip(col_abnormal, files_abnormal):
    FimgF,nbr_features = SIFT(img)
    cv2.imwrite("images/features/ABN-"+str(nbr_features)+"-"+filename,FimgF)
    print(filename+" , "+str(nbr_features))

