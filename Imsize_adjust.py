import cv2
import os

directory_init = "/Users/len/Desktop/HRI/YouCook/VideoFrames/copy"
directory_final = "/Users/len/Desktop/HRI/YouCook/VideoFrames/final"
total = 5640

for cnt in range(total):
    im1 = cv2.imread(os.path.join(directory_init,"{:0>6d}".format(cnt+1) + ".jpg"))
    print(os.path.join(directory_init,"{:0>6d}".format(cnt+1) + ".jpg"))

    #cv2.imshow('image1', im1)
    #cv2.waitKey(0)
    im2 = cv2.resize(im1, (500,300), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('image2', im2)
    #cv2.waitKey(0)
    cv2.imwrite(os.path.join(directory_final,"{:0>6d}".format(cnt+1) + ".jpg"), im2)
