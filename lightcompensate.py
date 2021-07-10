import cv2
import numpy as np

def unevenLightCompensate(img, blockSize, min_num,max_num):
    gray = img
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver
    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    # blockImage2[200:800,200:500]=0
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            if dst[i][j]<min_num:
                dst[i][j]=min_num
            elif dst[i][j]>max_num:
                dst[i][j]=max_num
    dst = dst.astype(np.uint8)
    # dst = cv2.GaussianBlur(dst, (3, 3), 0)

    return dst

path = "/home/pi/Desktop/face_recognition/code/image/asy.jpg"
img = cv2.imread(path)
img_norm = img/255.0  
img_gamma = np.power(img_norm,0.7)*255.0
img_gamma = img_gamma.astype(np.uint8)
img = cv2.cvtColor(img_gamma,cv2.COLOR_BGR2HSV)
cv2.imshow("gray",img)
dst2 = unevenLightCompensate(img[:,:,2], 100,0,255)
img[:,:,2] = dst2
img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
cv2.imshow("gray",img)
cv2.imwrite('/home/pi/Desktop/face_recognition/code/image/out_gamma_unevenLight.jpg', img)
