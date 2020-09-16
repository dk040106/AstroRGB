# 필요한 패키지를 가져온다.
# astropy.io.fits: fits 파일 입력
# matplotlib.pyplot: 실시간 이미지 시각화
# numpy: 데이터 처리
# PIL.Image: 이미지 파일 생성
# time: 보정 시간 측정

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# 이미지 파일이 위치한 폴더
target_dir = './examples/m51/'

# 각 색깔에 따른 가중치
r_w = 1
g_w = 0.7
b_w = 0.7

# 전체 밝기
t_w = 0.9

# 최소 경계값 (노이즈 제거)
MIN_THRES = 40

# 파일을 열어서 데이터를 가져온다.
r = fits.open(target_dir + 'r.fit')[0].data.astype(np.float64)
g = fits.open(target_dir + 'g.fit')[0].data.astype(np.float64)
b = fits.open(target_dir + 'b.fit')[0].data.astype(np.float64)


# 절댓값 오차 계산하기
def absError(a, b):
    return abs(a - b)


# 사진을 보정한 뒤 오차를 계산한다.
def error(r, g, b):
    errorCount = 0
    x_len = len(r)
    y_len = len(r[0])

    x_search = int(x_len / 2)
    y_search = int(y_len / 2)
    margin = 100

    for x in range(x_search - margin, x_search + margin):
        for y in range(y_search - margin, y_search + margin):
            errorSum = absError(
                r[x][y], g[x][y]) + absError(g[x][y], b[x][y]) + absError(b[x][y], r[x][y])
            errorCount += errorSum

    return errorCount


# 사진을 x, y만큼 이동하여 align한다.
def shift(r, g, b, x, y):
    # 0이면 오류가 나기 때문에 제외함
    if x == 0 or y == 0:
        return r, g, b

    if x < 0:
        r_1 = r[:x * 2, :]
        g_1 = g[-x: x, :]
        b_1 = b[-x * 2:, :]
    else:
        r_1 = r[x * 2:, :]
        g_1 = g[x: -x, :]
        b_1 = b[:-x * 2, :]

    if y < 0:
        r_2 = r_1[:, :y * 2]
        g_2 = g_1[:, -y: y]
        b_2 = b_1[:, -y * 2:]
    else:
        r_2 = r_1[:, y * 2:]
        g_2 = g_1[:, y:-y]
        b_2 = b_1[:, :-y * 2]

    return r_2, g_2, b_2


# 사진 하나에서 최소 오차를 align한다.
def alignImage(r, g, b):
    minX = 0
    minY = 0
    minError = error(r, g, b)

    for i in range(-10, 10):
        for j in range(-10, 10):
            r_s, g_s, b_s = shift(r, g, b, i, j)
            e = error(r_s, g_s, b_s)
            if(e < minError):
                minError = e
                minX = i
                minY = j

    return shift(r, g, b, minX, minY)


print("Aligning Images...")

# 시작 시간 저장
start = time.time()

# alignImage를 사용하여 이미지를 보정한다.
r, g, b = alignImage(r, g, b)

# 걸린 시간 출력
print("걸린 시간: {}초".format(int(time.time() - start)))

# 여기부터 재실행이 가능한 코드임.

# 가중치를 사용한 업데이트
r = r * r_w * t_w
g = g * g_w * t_w
b = b * b_w * t_w

# 255보다 큰 값들은 255로 대체한다.
r = np.where(r > 255, 255, r)
g = np.where(g > 255, 255, g)
b = np.where(b > 255, 255, b)

# 최소 경계값보다 작은 값들은 0으로 대체한다.
r = np.where(r < MIN_THRES, 0, r)
g = np.where(g < MIN_THRES, 0, g)
b = np.where(b < MIN_THRES, 0, b)

# 3개의 2차원 배열을 하나의 3차원 배열로 합친다.
print("Stacking Images...")
rgb = np.dstack((
    r.astype(np.uint8),
    g.astype(np.uint8),
    b.astype(np.uint8)
))


# 실시간 시각화를 위한 코드
# print("Showing Images...")
# plt.imshow(rgb)
# plt.show()

# 이미지 파일로 저장한다.
print("Saving Images...")
im = Image.fromarray(rgb)
im.save(target_dir + 'finished.png')
