import cv2
import numpy as np

def read_charuco_board(images, board, aruco_dict, debug=False):
    """
    Charuco base pose estimation.
    """
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        if debug:
            print(f"Processing image {decimator}")
        if (len(im[0].shape) == 4):
            gray = cv2.cvtColor(im[0].copy(), cv2.COLOR_BGR2GRAY)
        else:
            gray = im[0].copy()
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3:
                allCorners.append(res2[1])
                allIds.append(res2[2])
                if debug:
                    print("Apppending in the end")

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize

def read_checker_board(images, checker_data, debug=False):
    """
    Checker Board base pose estimation.
    """
    allCorners = []
    objectPointsArray = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    
    # create 3d pattern
    (rows, cols) = checker_data["cb_pattern"]
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objectPoints *= checker_data["cb_size"] 

    for im in images:
        gray = cv2.cvtColor(im[0], cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checker_data["cb_pattern"], None)
        # Make sure the chess board pattern was found in the image
        if ret:
            # Refine the corner position
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if(debug):
                if(not review_img(i, draw2(rgb, checker_data["cb_pattern"], corners, ret=True))):
                    continue
            # Add the object points and the image points to the arrays
            objectPointsArray.append(objectPoints)
            allCorners.append(corners)
            
    imsize = gray.shape
    return allCorners, objectPointsArray, imsize

def calibrate_checker_camera(allCorners, objectPointsArray, imsize, camera_matrix = np.eye(3), dist = np.zeros((5,1)), init_guess = False):
    """
    Calibrates the camera using the dected corners.
    """

    cameraMatrixInit = np.array([[ 600.,    0., imsize[0]/2.],
                                 [    0., 600., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    if init_guess:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, allCorners, imsize, camera_matrix, dist)
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, allCorners, imsize, cameraMatrixInit, np.zeros((5,1)))

    return ret, mtx, dist, rvecs, tvecs

def calibrate_charuco_camera(allCorners, allIds, imsize, board):
    """
    Calibrates the camera using the dected corners.
    """

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors