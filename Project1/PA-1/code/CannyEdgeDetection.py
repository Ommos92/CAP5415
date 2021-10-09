import numpy as np
import math
import cv2
import PIL
import matplotlib.pyplot as plt
import scipy.signal 
from skimage import data, filters

class cannyEdgeDetection():
    """
    Function Prototype:

    Class for implementation of Canny Edge Detection
    Algorithm overview:
    A multi-stage algorithm for edge detection

    1. Noise Reduction - First filter and smooth image using Gaussian Filter
    2. Find Intensity Gradient of Image - First derivative in horizontal 
    direction (Gx) and vertical direction (Gy). From these images two images we can find edge gradient and direction for each pixel

    Edge Gradient (G) = Sqrt((Gx)^2 + (Gy)^2)
    Angle(theta) = arctan(Gy/Gx)

    3. Non-Maximum Suppression - After getting gradient magnitude and direction, a full scan of image is done to remove any unwanted 
    pixels which many not be an edge. For every pixel, we check if it is a local maximum in its neighborhood in the driection of the gradient

    4. Hysteresis Thresholding  - This stage decides which edges are really edged and not an edge. For this we need two thresholds vals,
    minVal and Maxval. 

    """
    def __init__(self,imgPath,kSize,std,minVal,maxVal):
        self.imgPath = imgPath
        self.kSize = kSize
        self.std = std
        self.minVal = minVal
        self.maxVal = maxVal
        
    def GaussianKernel(self, OneDim = True, plot = False):
        """
        Input sigma value, and length of desired kernel
        returns a 1D or 2D Gaussian Kernel depending on OneDim True (1d) or False (2d)
        plot arg for showing kernel
        """
        #Center the array around zero
        ax = np.linspace(-(self.kSize-1)/2,(self.kSize-1)/2,self.kSize)
        Gaussian = np.exp(-(np.square(ax)/(2*np.square(self.std))))
        
        if OneDim == True :
            #One Dimensional Output
            Kernel = (1/np.sqrt(2*np.pi*self.std)) * Gaussian
            if plot == True:
                axis = plt.axes()
                axis.plot(ax, Kernel)
                plt.show()
            else:
                pass
            return Kernel
        else:
            #Two Dimensional Output
            Kernel = np.outer(Gaussian,Gaussian)
            if plot == True:
                plt.imshow(Kernel,interpolation='none')
                plt.show()
            else:
                pass
            return np.asmatrix(Kernel/ np.sum(Kernel))


    def loadImg(self, plot=True):
        """
        Return Mat object of input image, set plot = True to plot this function
        """
        
        I = cv2.imread(self.imgPath,0)
        if plot == True:
            print("Image has X {} pixels and Y {} pixels".format(I.shape[0],I.shape[1]))
            
            #Show original image
            #cv2.imshow('Original Image',I)
            #cv2.waitKey(0)
            plt.imshow(I, cmap='gray')
            plt.show()

        else:
            pass
        return I

    def oneDimMasks(self):
        """
        Return Gx and Gy Gaussian Masks
        
        By returning Gaussian Kernel Fct
        size of Gx is 1xn and Gy is calculated 
        by transposing from 1xn to nx1.
        """
        Gx = self.GaussianKernel()
        Gy = np.transpose(self.GaussianKernel())
        return Gx,Gy

    def convolutionOperation(self, padding = True, plotPad = False, plotResult = True):
        """
        Convolution operation to apply
        Kernel Filter for the Gaussian Blur Filter

        Convolution for 1D filter
        (a*v)[n] = Sum (a[m]*v[n-m]) from m = -inf to m = inf
        """

        # Load image and Gaussian kernels for x and y gradients
        I = self.loadImg()
        Gx,Gy = self.oneDimMasks()
        #reverse kernel for Convolution operation
        Gx = np.flip(Gx)
        Gy = np.flip(Gy)
        
        # Initialize zeros array for image size
        
            
        if padding == True:
            padwidth = int(self.kSize/2 - 1)
            I = np.pad(I,padwidth,mode='constant', constant_values = 0)
            XgaussFilter = np.zeros((I.shape[0], I.shape[1]))
            YgaussFilter = np.zeros((I.shape[0], I.shape[1]))
            
            if plotPad == True:
                plt.imshow(I)
                plt.show()
            
            #Perform Convolution on each direction X and Y
            for i in range(I.shape[0]):
                ##Need to do with own operation!!!
                XgaussFilter[i,:] = np.convolve(I[i,:], Gx, 'same')
            for j in range(I.shape[1]):
                #Redo with custom!!!!!!
                YgaussFilter[:,j] = np.convolve(I[:,j],Gy, 'same')
                if plotResult == True:
                    plt.imshow(XgaussFilter)
                    plt.show()
                    plt.imshow(YgaussFilter)
                    plt.show()
        else:
            #Redo!!!!
            XgaussFilter = np.zeros((I.shape[0], I.shape[1]))
            YgaussFilter = np.zeros((I.shape[0], I.shape[1]))
            #Perform Convolution on each direction X and Y
            for i in range(I.shape[0]):
                XgaussFilter[i,:] = np.convolve(I[i,:], Gx, 'same')
            for j in range(I.shape[1]):
                YgaussFilter[:,j] = np.convolve(I[:,j], Gy, 'same')
            if plotResult == True:
                    plt.imshow(XgaussFilter)
                    plt.show()
                    plt.imshow(YgaussFilter)
                    plt.show()
            return XgaussFilter, YgaussFilter

    def derivativeCalculation(self):
        """
        Using a sobel filter to convolve with the Gaussian
        Blurred image calculate the gradient and slope of gradient (theta)
        
        G = Sqrt(X^2 + Y^2)
        theta = arctan2(X/Y)

        """
        Xblur, Yblur = self.convolutionOperation(padding=False,plotPad=False,plotResult=False)
        #Create a sobel filter for this operation of the same size of kernel.
        #Kernel for both X and Y gradients
        Xgradient = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Ygradient = np.transpose(Xgradient)
        Yresult = scipy.signal.convolve2d(Yblur,Xgradient,'same')
        Xresult = scipy.signal.convolve2d(Xblur,Ygradient,'same')
        #plt.imshow(Xresult, cmap='gray')
        #plt.show()
        #plt.imshow(Yresult, cmap ='gray')
        #plt.show()

        Gradient = np.hypot(Xresult,Yresult)
        #Gradient normalization
        Gradient = (Gradient/Gradient.max()) * 255
        
        plt.imshow(Gradient, cmap ='gray')
        plt.show()
        #Calculate slope of gradient
        #theta = (180/math.pi) * (np.arctan2(Xresult,Yresult))
        #theta[theta < 0] += 180
        theta = np.arctan2(Xresult,Yresult)
        #plt.imshow(theta, cmap ='gray')
        #plt.show()
        return Gradient, theta
    
    def nonmaxSuppression(self, ploten = True):
        """
        Non-Max-Suppression:
        1. Create a matrix of zeros as the same size as the gradient output
        2. Identify the edge direction based on the angle value from the angle matrix
        3. Check if the pixel in the same direction has a higher intensity than the current pix
        4. Return the image 
        """
        
        G, theta = self.derivativeCalculation()
        #convert from rads to degrees and offset to +/- pi
        #theta = theta * 180. / np.pi
        theta[theta < 0] += np.pi/2
        
        #Create empty array for looping throguh image
        M, N = G.shape
        Z = np.zeros((M,N), dtype=np.int32)
        print("Row: {}, Column {}". format(M,N))

        #Loop through all points in the image
        for i in range(1,M-1):
            for j in range(1,N-1):
                #Find the neighboring pixels in the direction along theta vector
                #XY prime
                xp1 = math.cos(theta[i,j])
                yp1 = math.sin(theta[i,j])
                '''
                #This alogirthm does not account for 0,0 case 
                #interpolate closest pixel
                if  -0.5 < xp1 < 0.5:
                    col_xp1 = 0
                elif xp1 < -0.5:
                    col_xp1 = -1
                elif xp1 > 0.5:
                    col_xp1 = 1

                if -0.5 < yp1 < 0.5:
                    row_yp1 = 0
                elif yp1 <= 0.5:
                    row_yp1 = -1
                elif yp1 >= 0.5:
                    row_yp1 = 1
                '''

                if (xp1 < -0.5) and (yp1 < -0.5):
                    #South West
                    col_xp1 = -1
                    row_yp1 = -1
                
                elif (-0.5 <= xp1 <= 0.5) and (yp1 < -0.5):
                    #West
                    col_xp1 = -1
                    row_yp1 = 0

                elif (xp1 > 0.5) and (yp1 < -0.5):
                    #North West
                    col_xp1 = -1
                    row_yp1 = 1
                
                elif (xp1 < -0.5) and (-0.5 <= yp1 <= 0.5):
                    #South
                    col_xp1 = 0
                    row_yp1 = -1
                

                #Calculate the pix intensity for both points (odd function)
                try:
                    intensityP1 = G[i + col_xp1,j + row_yp1]
                    intensityP2 = G[i - col_xp1, j - row_yp1]
                    intensityOrigin = G[i,j]
                    
                    if (intensityOrigin >= intensityP1) and (intensityOrigin >= intensityP2):
                        Z[i,j] = G[i,j]
                    else:
                        Z[i,j] = 0
                except:
                    pass
                
        if(ploten == True):            
            plt.imshow(Z, cmap='gray')
            plt.show()
        return G


    def hysteresisThreshold(self,Low_Threshold,High_threshold):
        
        res = self.nonmaxSuppression()

        hyst = filters.apply_hysteresis_threshold(res, Low_Threshold, High_threshold)
        plt.imshow(hyst,cmap='gray')
        plt.show()
        return hyst
    
    
    '''
    These are examples for how it might be done
    '''
    def non_max_suppression(self):
        img, D = self.derivativeCalculation()
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        
        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255
                    
                #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0

                except IndexError as e:
                    pass
        plt.imshow(Z,cmap='gray')
        plt.show()
        return Z

    def threshold(self, lowThresholdRatio=0.1, highThresholdRatio=0.3):
        img = self.non_max_suppression()
        highThreshold = img.max() * highThresholdRatio;
        lowThreshold = highThreshold * lowThresholdRatio;
        
        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)
        
        weak = np.int32(25)
        strong = np.int32(255)
        
        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)
        
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        return (res, weak, strong)
    ##hysteresis example
    def hysteresis(self):
        img, weak, strong = self.threshold()
        plt.imshow(img)
        plt.plot()
        M, N = img.shape  
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        plt.imshow(img, cmap='gray')
        plt.show()
        return img
            
    

imgPath = r'C:/MastersCourses/gitWorkspace/CAP5415/CAP5415/Project1/images/img6.jpg'

minVal = 0
maxVal = 0
kSize = 5
std = 1.2
#plt.imshow(imgPath,cmap="gray")
cannyEdgeDetection(imgPath,kSize,std,minVal,maxVal).nonmaxSuppression()
cannyEdgeDetection(imgPath,kSize,std,minVal,maxVal).non_max_suppression()


