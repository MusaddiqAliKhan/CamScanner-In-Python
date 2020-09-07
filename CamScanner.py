import cv2
import numpy as np
import sys
import os


def camscanner(img):
    image= cv2.imread(img)
    if image is None:
        sys.exit('Cannot Find File')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 13) #Change the parameters here if you are dealing with loss of details in your image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    im = cv2.filter2D(thresh, -1, kernel)
    hist = cv2.equalizeHist(im)
    denoised = cv2.fastNlMeansDenoising(hist, 35, 31, 11) #Change the parameters here if you are dealing with image noise
    return denoised


def importing_images():
    
    folder = 'Enter the directory from where you want to import the images' #You can use sys.argv here if you want    
    d_folder = 'Enter the directory to where you want to save the scanned images'
    if not os.path.exists(d_folder):
        os.makedirs(d_folder)
    for i in os.listdir(folder):    
        orignal_img = folder+'\\'+i
        scanned_img = camscanner(orignal_img)
        clean_name = os.path.splitext(i)[0]
        cv2.imwrite(f'{d_folder}'+'\\'+f'{clean_name}.jpeg',scanned_img)
        

if __name__ == '__main__':
    importing_images()
