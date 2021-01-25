from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
import glob
import time

def detectBarcode(img):
    #경계값 추출
    edgeX=cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    edgeY=cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    subEdge=cv2.subtract(edgeX,edgeY)
    #영상 블러 및 쓰레숄딩
    blur = cv2.GaussianBlur(subEdge, (7, 9), 0)
    ret, threshold=cv2.threshold(blur,50, 180, cv2.THRESH_OTSU)
    #모폴리지 수행
    kernel= np.ones((5,69), np.uint8)
    close_morphol=cv2.morphologyEx(threshold,cv2.MORPH_CLOSE,kernel)
    do_erode = cv2.erode(close_morphol, kernel, iterations=5)
    do_dilate = cv2.dilate(do_erode, kernel, iterations=5)
    #연결요소 생성
    _, contours, _ =cv2.findContours(do_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cntr = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    else:
        cntr = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    subEdge_ = cv2.subtract(edgeY, edgeX)
    blur_ = cv2.GaussianBlur(subEdge_, (11, 9), 0)
    ret_, threshold_ = cv2.threshold(blur_, 100, 180, cv2.THRESH_OTSU)
    kernel_ = np.ones((63,5), np.uint8)
    close_morphol_ = cv2.morphologyEx(threshold_, cv2.MORPH_CLOSE, kernel_)
    do_erode_ = cv2.erode(close_morphol_, kernel_, iterations=4)
    do_dilate_ = cv2.dilate(do_erode_, kernel_, iterations=4)
    _, contours_, _ = cv2.findContours(do_dilate_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_:
        cntr_ = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    else:
        cntr_ = sorted(contours_, key=cv2.contourArea, reverse=True)[0]


    if len(cntr)>len(cntr_):
        (x, y, w, h) = cv2.boundingRect(cntr)
    else:
        (x, y, w, h) = cv2.boundingRect(cntr_)

    return (x, y, w, h)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True, help = "path to the dataset folder")
    ap.add_argument("-r", "--detectset", required = True, help = "path to the detectset folder")
    ap.add_argument("-f", "--detect", required = True, help = "path to the detect file")
    args = vars(ap.parse_args())
    
    dataset = args["dataset"]
    detectset = args["detectset"]
    detectfile = args["detect"]

    # 결과 영상 저장 폴더 존재 여부 확인
    if(not os.path.isdir(detectset)):
        os.mkdir(detectset)

    # 결과 영상 표시 여부
    verbose = False

    # 검출 결과 위치 저장을 위한 파일 생성
    f = open(detectfile, "wt", encoding="UTF-8")  # UT-8로 인코딩

    start=time.time()
    # 바코드 영상에 대한 바코드 영역 검출
    for imagePath in glob.glob(dataset + "/*.jpg"):
        print(imagePath, '처리중...')

        # 영상을 불러오고 그레이 스케일 영상으로 변환
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 바코드 검출
        (x,y,w,h)= detectBarcode(gray)

        # 바코드 영역 표시
        detectimg = cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)  # 이미지에 사각형 그리기

        # 결과 영상 저장
        loc1 = imagePath.rfind("\\")
        loc2 = imagePath.rfind(".")
        detectset=detectset+'\\'
        fname = detectset + imagePath[loc1 + 1: loc2] + '_res.jpg'
        cv2.imwrite(fname, detectimg)
        # 검출한 결과 위치 저장
        f.write(imagePath[loc1 + 1: loc2])
        f.write("\t")
        f.write(str(x))
        f.write("\t")
        f.write(str(y))
        f.write("\t")
        f.write(str(x+w))
        f.write("\t")
        f.write(str(y+h))
        f.write("\n")

        if verbose:
            cv2.imshow("image", image)
            cv2.waitKey(0)
    print("time : ", time.time()-start)