import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def makeGaussianKernel(size, sigma):
	center = (size-1)/2
	kernel = np.zeros((size, size))

	for x in range(size):
		for y in range(size):
			length = (x-center)**2 + (y-center)**2
			kernel[x][y] = np.exp(-length/(2*sigma**2))

	kernel /= kernel.sum()

	return kernel



def convolution2D(img, kernel, padding="reflection"):
	height = img.shape[0]
	width = img.shape[1]
	kernel_size = kernel.shape[0]
	tmp_img = np.zeros((height+kernel_size-1,width+kernel_size-1))
	result_img = np.zeros((height,width))
	half_kernel_size = int((kernel_size-1)/2)

	if padding=="reflection":
		tmp_img[half_kernel_size:half_kernel_size+height, half_kernel_size:half_kernel_size+width] = img[:, :]
		tmp_img[:half_kernel_size, half_kernel_size:half_kernel_size+width] = img[half_kernel_size:0:-1, :] #top
		tmp_img[half_kernel_size+height:, half_kernel_size:half_kernel_size+width] = img[height-2:height-half_kernel_size-2:-1, :] #bottom
		tmp_img[half_kernel_size:half_kernel_size+height, :half_kernel_size] = img[:, half_kernel_size:0:-1] #left
		tmp_img[half_kernel_size:half_kernel_size+height, half_kernel_size+width:] = img[:, width-2:width-half_kernel_size-2:-1] #right
		tmp_img[:half_kernel_size, :half_kernel_size] = img[half_kernel_size:0:-1, half_kernel_size:0:-1] #top left
		tmp_img[:half_kernel_size, half_kernel_size+width:] = img[half_kernel_size:0:-1, width-2:width-half_kernel_size-2:-1] #top right
		tmp_img[half_kernel_size+height:, :half_kernel_size] = img[height-2:height-half_kernel_size-2:-1, half_kernel_size:0:-1] #bottom left
		tmp_img[half_kernel_size+height:, half_kernel_size+width:] = img[height-2:height-half_kernel_size-2:-1, width-2:width-half_kernel_size-2:-1] #bottom right

	elif padding=="zero":
		pass
	else:
		raise ValueError("unexpected padding value")

	for x in range(height):
		for y in range(width):
			convol_sum = 0
			for i in range(kernel_size):
				for j in range(kernel_size):
					convol_sum += tmp_img[x+i,y+j]*kernel[i,j]
			result_img[x,y] = convol_sum

	return result_img



def harrisCornerDetection(img, box_rad, k):
	height = img.shape[0]
	width = img.shape[1]
	diff_x_kernel = np.array(
		[[-1, -2, -1],
		[0, 0, 0],
		[1, 2, 1]]
		)
	diff_y_kernel = np.array(
		[[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]]
		)
	diff_x = convolution2D(img, diff_x_kernel)
	diff_y = convolution2D(img, diff_y_kernel)

	mask = np.full((height,width), 0)

	for x in range(height-box_rad*2):
		for y in range(width-box_rad*2):
			Ix2 = 0
			Iy2 = 0
			Ixy = 0

			for i in range(box_rad*2+1):
				for j in range(box_rad*2+1):
					Ix2 += diff_x[x+i,y+j]**2
					Iy2 += diff_y[x+i,y+j]**2
					Ixy += diff_x[x+i,y+j]*diff_y[x+i,y+j]

			det = Ix2*Iy2 - Ixy**2
			trace = Ix2 + Iy2
			R = det - k*(trace**2)

			mask[x+box_rad,y+box_rad] = R/1000

	return mask

def nonMaximumSuppression(image, window_rad):
	img = image.copy()
	height = img.shape[0]
	width = img.shape[1]

	for x in range(height):
		for y in range(width):
			if img[x,y] != 0:
				max_R = 0

				for i in range(2*window_rad+1):
					for j in range(2*window_rad+1):
						if x-window_rad+i>=0 and x-window_rad+i<height and y-window_rad+j>=0 and y-window_rad+j<width:
							if img[x-window_rad+i,y-window_rad+j] > max_R:
								max_R = img[x-window_rad+i,y-window_rad+j]


				for i in range(2*window_rad+1):
					for j in range(2*window_rad+1):
						if x-window_rad+i>=0 and x-window_rad+i<height and y-window_rad+j>=0 and y-window_rad+j<width:
							if img[x-window_rad+i,y-window_rad+j] < max_R:
								img[x-window_rad+i,y-window_rad+j] = 0

	result = []
	for x in range(height):
		for y in range(width):
			if img[x,y]!=0:
				result.append((x,y))


	return result



def findCorners(image):
	print("making gaussian kernel...")
	kernel = makeGaussianKernel(7, 7)

	print("executing convolution...")
	blured_image = convolution2D(image, kernel)

	print("executing harris corner detection...")
	mask = harrisCornerDetection(blured_image, 3, 0.04)

	mask_max = 0
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			if mask[x,y] > mask_max:
				mask_max = mask[x,y]



	image_tmp_1 = image.copy()
	for x in range(image_tmp_1.shape[0]):
		for y in range(image_tmp_1.shape[1]):
			if mask[x,y] < 0.01*mask_max:
				mask[x,y] = 0

	cv2.imwrite("./harris.png", image_tmp_1)

	print("executing NMS...")
	corners = nonMaximumSuppression(mask, 30)

	image_tmp_2 = image.copy()
	for i in corners:
		image_tmp_2[i[0],i[1]] = 255

	cv2.imwrite("./NMS.png", image_tmp_2)

	return corners

def findCorrespondence(img1, corners1, img2, corners2, window_rad):
	correspond_min_point = []
	for cord1 in corners1:
		x1 = cord1[0]
		y1 = cord1[1]
		window1 = img1[x1-window_rad:x1+window_rad+1,y1-window_rad:y1+window_rad+1]

		correspond_min = float('inf')
		flag = False

		for cord2 in corners2:
			x2 = cord2[0]
			y2 = cord2[1]
			window2 = img2[x2-window_rad:x2+window_rad+1,y2-window_rad:y2+window_rad+1]

			correspond = 0
			if window1.shape[0]==2*window_rad+1 and window1.shape[1]==2*window_rad+1 and window2.shape[0]==2*window_rad+1 and window2.shape[1]==2*window_rad+1:
				for i in range(2*window_rad+1):
					for j in range(2*window_rad+1):
						correspond += (int(window1[i,j]) - int(window2[i,j]))**2

				if correspond < correspond_min:
					correspond_min = correspond
					correspond_point = [x2,y2]
					flag = True

		if flag:
			point_tmp = ((x1,y1),correspond_point)
			correspond_min_point.append(point_tmp)

	return correspond_min_point

def ransac(correspond, point_num, num_repeat, threshold, img1_width):
    best_inlier = []
    epsilon = 1e-6

    for repeat in range(num_repeat):
        # 1. 랜덤 샘플링
        sampled_points = random.sample(correspond, point_num)

        inclinations = []
        for (y1, x1), (y2, x2) in sampled_points:
            x2 += img1_width
            if abs(x1 - x2) > epsilon:
                inclinations.append(np.arctan((y1 - y2) / float(x1 - x2)))

        if not inclinations: 
            continue

        inclination = np.median(inclinations)

        inliers = []
        for (y1, x1), (y2, x2) in correspond:
            x2 += img1_width
            if abs(x1 - x2) < epsilon:  # 오버플로우 방지
                continue
            tmp_inclination = np.arctan((y1 - y2) / float(x1 - x2))
            if abs(tmp_inclination - inclination) < threshold:
                inliers.append(((y1, x1), (y2, x2-img1_width)))

        if len(inliers) > len(best_inlier):
            best_inlier = inliers

    return best_inlier

def findHomography(correspond, num_repeat):
	error_min = float('inf')
	result_H = np.zeros((3,3))
	for j in range(num_repeat):
		mat = np.zeros((8,9))
		tmp = random.sample(correspond, 4)
		for i in range(4):
			(x1, y1), (x2, y2) = tmp[i]
			row1 = [x2, y2, 1, 0, 0, 0, -x2*x1, -y2*x1, -x1]
			row2 = [0, 0, 0, x2, y2, 1, -x2*y1, -y2*y1, -y1]
			mat[i*2,:] = row1[:]
			mat[i*2+1,:] = row2[:]

		U, s, VT = np.linalg.svd(mat)

		H = VT[-1].reshape(3, 3)

		H = H / H[2, 2]

		error = 0

		for cor in correspond:
			(x1, y1), (x2, y2) = cor
			x_pred = H[0][0]*x2 + H[0][1]*y2 + H[0][2]
			y_pred = H[1][0]*x2 + H[1][1]*y2 + H[1][2]
			z = H[2][0]*x1 + H[2][1]*y1 + H[2][2]

			x_pred /= z
			y_pred /= z

			error += (x_pred-x1)**2 + (y_pred-y1)**2

		if error < error_min:
			error_min = error
			result_H = H


	return result_H

def stitching(img_src, img_dst, H, size):
    result = img_dst
    # result[:img_dst.shape[0],:img_dst.shape[1],:] = img_dst[:,:,:]

    H = np.linalg.inv(H)
    
    for x in range(size[0]):
        for y in range(size[1]):
            src_x = H[0][0] * x + H[0][1] * y + H[0][2]
            src_y = H[1][0] * x + H[1][1] * y + H[1][2]
            z = H[2][0] * x + H[2][1] * y + H[2][2]

            if z == 0:
                continue
            
            src_x /= z
            src_y /= z

            if 0 <= src_x < img_src.shape[0] - 1 and 0 <= src_y < img_src.shape[1] - 1:
                x1, y1 = int(src_x), int(src_y)
                a, b = src_x - x1, src_y - y1

                # Ensure indices are within bounds
                if x1 >= 0 and y1 >= 0 and x1 + 1 < img_src.shape[0] and y1 + 1 < img_src.shape[1]:
                    for c in range(img_src.shape[2]):
                        f00 = img_src[x1, y1, c]
                        f01 = img_src[x1, y1 + 1, c]
                        f10 = img_src[x1 + 1, y1, c]
                        f11 = img_src[x1 + 1, y1 + 1, c]

                        value = (
                            f00 * (1 - a) * (1 - b) +
                            f10 * a * (1 - b) +
                            f01 * (1 - a) * b +
                            f11 * a * b
                        )

                        result[x, y, c] = np.clip(round(value), 0, 255)

    return result


def mult_matirx(M1, M2):
	result = np.zeros((3,3))

	for i in range(3):
		for j in range(3):
			sumation = 0
			for k in range(3):
				sumation += M1[i,k]*M2[k,j]

			result[i,j] = sumation
	result /= result[2,2]

	return result


print("loading data...")
image1 = cv2.imread('./testimg1.jpg')
image2 = cv2.imread('./testimg2.jpg')
image3 = cv2.imread('./testimg3.jpg')
# image4 = cv2.imread('./testimg4.jpg')
# image5 = cv2.imread('./testimg5.jpg')
# image6 = cv2.imread('./testimg6.jpg')
# image7 = cv2.imread('./testimg7.jpg')
# image8 = cv2.imread('./testimg8.jpg')
# image9 = cv2.imread('./testimg9.jpg')
# image10 = cv2.imread('./testimg10.jpg')

image1_gray = cv2.imread('./testimg1.jpg', cv2.IMREAD_GRAYSCALE)
image2_gray = cv2.imread('./testimg2.jpg', cv2.IMREAD_GRAYSCALE)
image3_gray = cv2.imread('./testimg3.jpg', cv2.IMREAD_GRAYSCALE)
# image4_gray = cv2.imread('./testimg4.jpg', cv2.IMREAD_GRAYSCALE)
# image5_gray = cv2.imread('./testimg5.jpg', cv2.IMREAD_GRAYSCALE)
# image6_gray = cv2.imread('./testimg6.jpg', cv2.IMREAD_GRAYSCALE)
# image7_gray = cv2.imread('./testimg7.jpg', cv2.IMREAD_GRAYSCALE)
# image8_gray = cv2.imread('./testimg8.jpg', cv2.IMREAD_GRAYSCALE)
# image9_gray = cv2.imread('./testimg9.jpg', cv2.IMREAD_GRAYSCALE)
# image10_gray = cv2.imread('./testimg10.jpg', cv2.IMREAD_GRAYSCALE)



# print('\nimage1\n')
# corners1 = findCorners(image1_gray)
# print('\nimage2\n')
# corners2 = findCorners(image2_gray)
# print('\nimage3\n')
# corners3 = findCorners(image3_gray)
# print('\nimage4\n')
# corners4 = findCorners(image4_gray)
# print('\nimage5\n')
# corners5 = findCorners(image5_gray)
# print('\nimage6\n')
# corners6 = findCorners(image6_gray)
# print('\nimage7\n')
# corners7 = findCorners(image7_gray)
# print('\nimage8\n')
# corners8 = findCorners(image8_gray)
# print('\nimage9\n')
# corners9 = findCorners(image9_gray)
# print('\nimage10\n')
# corners10 = findCorners(image10_gray)

# correspond12 = findCorrespondence(image1_gray, corners1, image2_gray, corners2, 70)
# correspond23 = findCorrespondence(image2_gray, corners2, image3_gray, corners3, 70)
# correspond34 = findCorrespondence(image3_gray, corners3, image4_gray, corners4, 70)
# correspond45 = findCorrespondence(image4_gray, corners4, image5_gray, corners5, 70)
# correspond56 = findCorrespondence(image5_gray, corners5, image6_gray, corners6, 70)
# correspond67 = findCorrespondence(image6_gray, corners6, image7_gray, corners7, 70)
# correspond78 = findCorrespondence(image7_gray, corners7, image8_gray, corners8, 70)
# correspond89 = findCorrespondence(image8_gray, corners8, image9_gray, corners9, 70)
# correspond910 = findCorrespondence(image9_gray, corners9, image10_gray, corners10, 70)

# correspond12 = ransac(correspond12, 5, 10, np.pi/25, image1_gray.shape[1])
# correspond23 = ransac(correspond23, 5, 10, np.pi/25, image2_gray.shape[1])
# correspond34 = ransac(correspond34, 5, 10, np.pi/25, image3_gray.shape[1])
# correspond45 = ransac(correspond45, 5, 10, np.pi/25, image4_gray.shape[1])
# correspond56 = ransac(correspond56, 5, 10, np.pi/25, image5_gray.shape[1])
# correspond67 = ransac(correspond67, 5, 10, np.pi/25, image6_gray.shape[1])
# correspond78 = ransac(correspond78, 5, 10, np.pi/25, image7_gray.shape[1])
# correspond89 = ransac(correspond89, 5, 10, np.pi/25, image8_gray.shape[1])
# correspond910 = ransac(correspond910, 5, 10, np.pi/25, image9_gray.shape[1])
# print(correspond12)
# print(correspond23)
correspond12 = [((99, 1089), (73, 770)), ((203, 351), (83, 2303)), ((204, 505), (73, 770)), ((330, 1432), (325, 1139)), ((350, 1096), (344, 776)), ((368, 1368), (363, 1074)), ((374, 811), (370, 464)), ((385, 1082), (381, 768)), ((398, 526), (393, 112)), ((404, 773), (403, 422)), ((404, 892), (403, 557)), ((415, 563), (414, 168)), ((435, 926), (436, 595)), ((437, 892), (439, 557)), ((480, 860), (488, 522)), ((502, 1433), (505, 1145)), ((504, 1470), (515, 1195)), ((507, 1861), (506, 1578)), ((510, 1544), (505, 1250)), ((525, 877), (536, 541)), ((563, 892), (577, 560)), ((571, 1858), (571, 1577)), ((711, 768), (744, 424)), ((745, 752), (782, 404)), ((788, 73), (571, 1577)), ((839, 352), (577, 560)), ((875, 92), (506, 1578)), ((879, 766), (931, 425)), ((910, 738), (964, 393)), ((911, 769), (968, 425)), ((1033, 854), (1096, 529)), ((1072, 854), (1139, 530)), ((1086, 1068), (1141, 770)), ((1131, 1080), (1188, 784)), ((1154, 1348), (1195, 1072)), ((1201, 1347), (1244, 1072))]
correspond23 = [((73, 770), (402, 1830)), ((83, 2303), (104, 1758)), ((173, 82), (402, 1830)), ((325, 1139), (307, 680)), ((344, 776), (323, 285)), ((363, 1074), (346, 613)), ((370, 464), (104, 1758)), ((381, 768), (361, 291)), ((393, 112), (402, 1830)), ((396, 522), (104, 1758)), ((400, 595), (380, 74)), ((401, 462), (361, 291)), ((403, 422), (346, 613)), ((403, 557), (380, 74)), ((414, 168), (402, 1830)), ((436, 595), (419, 76)), ((439, 557), (419, 76)), ((488, 522), (104, 1758)), ((505, 1145), (500, 690)), ((505, 1250), (490, 804)), ((506, 1578), (490, 1131)), ((515, 1195), (493, 752)), ((536, 541), (402, 1830)), ((567, 525), (402, 1830)), ((571, 1577), (553, 1132)), ((577, 560), (402, 1830)), ((748, 389), (671, 2027)), ((782, 404), (769, 2026)), ((960, 1889), (913, 1432)), ((964, 393), (769, 2026)), ((968, 425), (1243, 636)), ((1141, 770), (1165, 302)), ((1188, 784), (1214, 317)), ((1195, 1072), (1193, 634)), ((1244, 1072), (1243, 636)), ((1352, 1946), (1275, 1495))]
print('finding homography')
H12 = findHomography(correspond12, 200)
H23 = findHomography(correspond23, 200)
print(H12)
# H34 = findHomography(correspond34, 50)
# H45 = findHomography(correspond45, 50)
# H56 = findHomography(correspond56, 50)
# H67 = findHomography(correspond67, 50)
# H78 = findHomography(correspond78, 50)
# H89 = findHomography(correspond89, 50)
# H910 = findHomography(correspond910, 50)


print('starting stitch')
print('\nimage1-2\n')
size = (3000,15000,3)
base = np.zeros(size)
base[:image2.shape[0],:image2.shape[1],:] = image2[:,:,:]
result2 = stitching(image3, base, H23, size)
cv2.imwrite("./tmp.jpg", result2)
# H123 = mult_matirx(H23,H12)
# print('\nimage2-3\n')
# result3 = stitching(image3, base, H23, size)
# H1234 = mult_matirx(H34,H123)
# print('\nimage3-4\n')
# result4 = stitching(image4, result3, H1234, size)
# H12345 = mult_matirx(H45,H1234)
# print('\nimage4-5\n')
# result5 = stitching(image5, result4, H12345, size)
# H123456 = mult_matirx(H56,H12345)
# print('\nimage5-6\n')
# result6 = stitching(image6, result5, H123456, size)
# H1234567 = mult_matirx(H67,H123456)
# print('\nimage6-7\n')
# result7 = stitching(image7, result6, H1234567, size)
# H12345678 = mult_matirx(H78,H1234567)
# print('\nimage7-8\n')
# result8 = stitching(image8, result7, H12345678, size)
# H123456789 = mult_matirx(H89,H12345678)
# print('\nimage8-9\n')
# result9 = stitching(image9, result8, H123456789, size)
# H12345678910 = mult_matirx(H910,H123456789)
# print('\nimage9-10\n')
# result10 = stitching(image10, result9, H12345678910, size)

cv2.imwrite("./result.jpg", result2)