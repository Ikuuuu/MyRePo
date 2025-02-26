import cv2
import numpy as np
import os

# 현재 스크립트의 디렉토리 경로를 가져오기
script_dir = os.path.dirname(__file__)

# 이미지 파일의 절대 경로 생성
image_path = os.path.join(script_dir, 'Lenna.png')

# 이미지를 읽기
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 이미지 읽기 확인
if image is None:
    print("이미지를 읽을 수 없습니다. 경로를 확인하세요.")
else:
    # 평균값 필터 적용
    avg_filter = cv2.blur(image, (5, 5))

    # 샤프닝 필터 적용
    kernel_sharpening = np.array([[-1, -1, -1], 
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)

    # 라플라시안 필터 적용
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # 결과 출력
    cv2.imshow('Original', image)
    cv2.imshow('Average Filter', avg_filter)
    cv2.imshow('Sharpening Filter', sharpened)
    cv2.imshow('Laplacian Filter', laplacian)

    cv2.waitKey(0)  # 키 입력을 대기

    cv2.destroyAllWindows()  # 모든 창 닫기