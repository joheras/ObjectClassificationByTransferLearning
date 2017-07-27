import cv2
import numpy as np
import math

def describe(image,p_segments):
    (h, w) = image.shape[:2]
    control = image[0:h, 0:w / 2]
    hC = h
    wC = w/2
    segments = 2**p_segments

    # Mask to only keep the centre
    mask = np.zeros(control.shape[:2], dtype="uint8")

    (h, w) = control.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    masks = [mask.copy() for i in range(0, 8*segments)]
    # Generating the different annulus masks
    for i in range(0, 8*segments):
        cv2.circle(masks[i], (cX, cY), min(90 - 10 * (i % 8), control.shape[1]) / 2, 255, -1)
        cv2.circle(masks[i], (cX, cY), min(80 - 10 * (i % 8), control.shape[1]) / 2, 0, -1)


    if (p_segments == 2):
        points = np.array([[cX,cY],[cX,0],[0,0],[0,h],[w,h],[w,cY],[cX,cY]], np.int32)
        points = points.reshape((-1,1,2))
        for i in range(0, 8):
            cv2.fillConvexPoly(masks[i],points,0)
    else:
        for k in range(0,2**(p_segments - 2)):
            alpha = (math.pi/2**(p_segments - 1))*(k+1)
            beta = (math.pi/2**(p_segments - 1))*k
            if alpha <= math.pi/4:
                points = np.array([[cX, cY], [w, h/2-w/2*math.tan(alpha)], [w,0], [0, 0], [0, h], [w, h],
                                   [w, h/2-w/2*math.tan(beta)], [cX, cY]], np.int32)
                points = points.reshape((-1, 1, 2))
                points2 = np.array([[cX, cY], [w,cY], [w, h/2-w/2*math.tan(beta)], [cX, cY]], np.int32)
                points2 = points2.reshape((-1, 1, 2))
                for i in range(0, 8):
                    cv2.fillConvexPoly(masks[8*k+i], points, 0)
                    cv2.fillConvexPoly(masks[8 * k + i], points2, 0)


            else:
                points = np.array([[cX, cY], [cX+(h/2)/math.tan(alpha),0], [0, 0], [0, h], [w, h], [w,0],
                                   [cX+ (h / 2)/math.tan(beta),0], [cX, cY]], np.int32)
                points = points.reshape((-1, 1, 2))
                points2 = np.array([[cX, cY], [cX+ (h / 2)/math.tan(beta),0], [w, 0],[w,cY], [cX, cY]], np.int32)
                points2 = points2.reshape((-1, 1, 2))
                for i in range(0, 8):
                    cv2.fillConvexPoly(masks[8*k+i], points, 0)
                    cv2.fillConvexPoly(masks[8 * k + i], points2, 0)

    M90 = cv2.getRotationMatrix2D((cX,cY),90,1.0)
    M180 = cv2.getRotationMatrix2D((cX,cY),180,1.0)
    M270 = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)

    for i in range(0,8*(2**(p_segments-2))):
        masks[8*(2**(p_segments-2))+i]= cv2.warpAffine(masks[i],M90,(w,h))
        masks[2* 8 * (2 ** (p_segments - 2)) + i] =cv2.warpAffine(masks[i],M180,(w,h))
        masks[3* 8 * (2 ** (p_segments - 2)) + i] = cv2.warpAffine(masks[i],M270,(w,h))

    rows = segments
    cols = 8
    figure = np.zeros((rows*hC,cols*wC))
    for i in range(rows):
        for j in range(cols):

            figure[i*hC:(i+1)*hC,j*wC:(j+1)*wC] = masks[cols*i+j]

    cv2.imwrite("test.jpg",figure)



im = cv2.imread('/home/joheras/Escritorio/Research/Fungi/FungiImagesWithControl/-/ad71aem-ad71aem_3.jpg')
describe(im,4)
