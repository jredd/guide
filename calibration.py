import numpy as np
import cv2

# Define the chess board rows and columns
rows = 9
cols = 6

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((rows*cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.
obj_points = []  # 3d point in real world space
img_points_left = []  # 2d points in image plane.
img_points_right = []  # 2d points in image plane.


capture_left = cv2.VideoCapture('http://10.0.1.4:8081/img')
capture_right = cv2.VideoCapture('http://10.0.1.5:8081/img')
count = 0
tot_error_left = 0
tot_error_right = 0

while True:

    # # Retrieve the latest image from the webcam
    # rc, fullSizeBaseImage = capture.read()
    rc, img_left = capture_left.read()
    rc2, img_right = capture_right.read()
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (rows, cols), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (rows, cols), None)

    # Make sure the chess board pattern was found in the image
    if ret_left and ret_right:
        obj_points.append(objp)

        cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        img_points_left.append(corners_left)
        img_points_right.append(corners_right)

        # Draw and display the corners
        cv2.drawChessboardCorners(img_left, (7, 6), corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, (7, 6), corners_right, ret_right)
        cv2.imshow('img_left', img_left)
        cv2.imshow('img_right', img_right)

        ret_l, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            obj_points,
            img_points_left,
            gray_left.shape[::-1],
            None, None
        )

        ret_r, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            obj_points,
            img_points_left,
            gray_left.shape[::-1],
            None, None
        )
        image_size = gray_left.shape[::-1]
        (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
            obj_points, img_points_left, img_points_right,
            mtx_left, dist_left,
            mtx_right, dist_right,
            image_size, None, None, None, None,
            cv2.CALIB_FIX_INTRINSIC, criteria)

        (leftRectification, rightRectification, leftProjection, rightProjection,
         dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
            mtx_left, dist_left,
            mtx_right, dist_right,
            image_size, rotationMatrix, translationVector,
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY,
            # OPTIMIZE_ALPHA,
        )

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, leftRectification,
            leftProjection, image_size, cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, rightRectification,
            rightProjection, image_size, cv2.CV_32FC1)

        np.savez_compressed('calibration_data/stereo_cams4-5.npz', imageSize=image_size,
                            leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
                            rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)


        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # np.savez('calibration_data/calib_cam.5.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

        # # np.load
        # # img2 = cv2.imread('capture/cam.4.jpg')
        # img2 = cv2.imread('capture/cam.5.jpg')
        # h, w = img2.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        # cv2.imshow('calibImg', dst)
        # # cv2.imwrite('calibresult.png', dst)

        mean_error = 0
        for i in range(len(obj_points)):
            imgpoints2_left, _ = cv2.projectPoints(obj_points[i], rvecs_left[i], tvecs_left[i], mtx_left, dist_left)
            imgpoints2_right, _ = cv2.projectPoints(obj_points[i], rvecs_right[i], tvecs_right[i], mtx_right, dist_right)
            error_left = cv2.norm(img_points_left[i], imgpoints2_left, cv2.NORM_L2) / len(imgpoints2_left)
            error_right = cv2.norm(img_points_right[i], imgpoints2_right, cv2.NORM_L2) / len(imgpoints2_right)
            tot_error_left += error_left
            tot_error_right += error_right

        print("total error left: ", mean_error / len(obj_points))
        print("total error right: ", mean_error / len(obj_points))
        count += 1
        print('frame count:', count)







        # # Refine the corner position
        # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #
        # # Add the object points and the image points to the arrays
        # objectPointsArray.append(objectPoints)
        # imgPointsArray.append(corners)
        #
        # # Draw the corners on the image
        # cv2.drawChessboardCorners(fullSizeBaseImage, (rows, cols), corners, ret)
        #
        #
        # # Display the image
        # cv2.imshow('chess original', fullSizeBaseImage)
        # cv2.waitKey(500)
        #
        # # Calibrate the camera and save the results
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
        # np.savez('calibration_data/calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        #
        # # Print the camera calibration error
        # error = 0
        #
        # for i in range(len(objectPointsArray)):
        #     imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
        #     error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)
        #
        # print("Total error: ", error / len(objectPointsArray))
        # count += 1
        # print('img:', count)
        # # Load one of the test images
        # img = cv2.imread('checker1.png')
        # h, w = img.shape[:2]
        #
        # # Obtain the new camera matrix and undistort the image
        # newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
        #
        # # Crop the undistorted image
        # # x, y, w, h = roi
        # # undistortedImg = undistortedImg[y:y + h, x:x + w]
        #
        # # Display the final result

        # cv2.imshow('chess board', np.hstack((img, undistortedImg)))
        # cv2.waitKey(0)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

        break

capture.release()
cv2.destroyAllWindows()