import numpy as np
np.set_printoptions(suppress=True)
import pickle
import sys, os
import time
import cv2
import quaternion
from IPython import display
from IPython.display import clear_output
import matplotlib.pyplot as plt
import geo_utils

def estimate_cam_pose_checker_old(rgb, K, dist,  cb_pattern, pattern_3d, dist_coeff, criteria):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
    #Calculate corner Points
    points_2d, list_points_2d = [], []
    ret, points_2d = cv2.findChessboardCorners(gray, cb_pattern, None)

    if not ret:
        print('No marker found... skipping this sample: '  + '.')
        return None
    corner_img = None   
    
    points_2d = cv2.cornerSubPix(gray, points_2d, (11,11), (-1,-1), criteria)
    _, rVec, tVec = cv2.solvePnP(np.transpose(pattern_3d), points_2d, K, dist)
    
    points_2d_ = []
    
    for pts in points_2d:
        points_2d_.append(pts[0])
        
    list_points_2d.append(np.array(points_2d_).transpose()) 
    t_obj= np.array([tVec[0][0], tVec[1][0], tVec[2][0]])
    R_obj = cv2.Rodrigues(rVec)[0]
    
    hom_obj = geo_utils.create_transform(R=R_obj, t = t_obj)
    
    return t_obj, R_obj, hom_obj, points_2d, points_2d_, list_points_2d, corner_img

def tvec_rvec_to_homTF(t_obj, p_rvec):
    R_obj = cv2.Rodrigues(p_rvec)[0]
    hom_obj = geo_utils.create_transform(R=R_obj, t = t_obj)
    return hom_obj

def review_img(rgb):
    
    #display.Image(rgb)
    plt.figure(figsize = (20,20))
    plt.imshow(rgb)
    plt.show()
    A=input('is this image good? (y)')

    if(A=='' or A == 'y'):
        print('good soup')
        soup = True
    else:
        print('bad soup')
        soup = False

    time.sleep(.5)
    clear_output(wait=True)
    
    return soup

def estimate_cam_pose_checker(rgbs, K, dist, checker_data, dist_coeff, criteria, subpixel_accuracy = True, debug=True):
    
    assert rgbs.shape != (0, )
    
    # generate pattern_3d
    pattern_3d = np.zeros((np.prod(checker_data["cb_pattern"]), 3), np.float32)
    pattern_3d[:, :2] = np.indices(checker_data["cb_pattern"]).T.reshape(-1, 2)
    pattern_3d *= checker_data["cb_size"]
    pattern_3d = np.transpose(pattern_3d)

    
    p_tvec_median, p_rvec_median, p_rvec_all, p_tvec_all, hom_TF, corner_img_all, points_2d_all= [],  [],  [], [],  [], [], []
    
    for i, rgb in enumerate(rgbs):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)
        if (len(rgbs.shape) == 4):
            gray = cv2.cvtColor(rgb.copy(), cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb.copy()
            
        # detect chessboard corners
        corners, list_points_2d = [], []
        ret, corners = cv2.findChessboardCorners(gray, checker_data["cb_pattern"], None)

        if not ret:
            if debug:
                print('No marker found... skipping this sample: '  + '.')
            continue
            
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        _, rVec, tVec = cv2.solvePnP(np.transpose(pattern_3d), corners, K, dist)
    
        # transform points for pose graph otpim
        corners_ = []
        for pts in corners:
            corners_.append(pts[0])
        list_points_2d.append(np.array(corners_).transpose()) 
        
        p_tvec= np.array([tVec[0][0], tVec[1][0], tVec[2][0]])
        p_rvec = np.array(rVec)
        
        # append to lists
        p_rvec_all.append(p_rvec)
        p_tvec_all.append(p_tvec)
        points_2d_all.append(list_points_2d)
        
        corner_img_all = []
        if debug:
            corner_img = cv2.drawChessboardCorners(rgb.copy(), checker_data["cb_pattern"], corners, True)
            corner_img_all.append(corner_img)
   
    if(len(p_tvec_all) > 0 and len(p_rvec_all) > 0):
        p_tvec_median = np.median(p_tvec_all, axis = 0)       
        p_rvec_median = np.median(p_rvec_all, axis = 0)
        hom_TF = tvec_rvec_to_homTF(p_tvec_median, p_rvec_median)
        
    return p_tvec_median, p_rvec_median, hom_TF, corner_img_all, corner_img_all
    

def estimate_cam_pose_charuco(rgbs, camera_matrix, dist_coeff, board, aruco_dict, subpixel_accuracy = True, debug=True):
    
    assert rgbs.shape != (0, )
    
    p_rvec_all, p_tvec_all, imgs_all, corners_all, c_corner_all, points_2d_all= [],  [],  [],  [],  [],  []
    
    for i, rgb in enumerate(rgbs):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if (len(rgbs.shape) == 4):
            frame = cv2.cvtColor(rgb.copy(), cv2.COLOR_BGR2GRAY)
        else:
            frame = rgb.copy()
        corners, ids, rejected_points = cv2.aruco.detectMarkers(frame, aruco_dict)

        if len(corners) != len(ids) or len(corners) == 0:
            return None

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict)

        if(len(corners)>0 and subpixel_accuracy):
            if debug:
                print('Using Subpixel accuracy')
            #corners = cv2.cornerSubPix(frame, corners, (3,3), (-1,-1), criteria)
            for corner in corners:
                corner = cv2.cornerSubPix(frame, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)


        if len(corners) != len(ids) or len(corners) == 0:
            return None


        ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners,
                                                                    ids,
                                                                    frame,
                                                                    board)

        ret, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners,
                                                                c_ids,
                                                                board,
                                                                camera_matrix,
                                                                dist_coeff,
                                                                np.zeros((3,1)),
                                                                np.zeros((3,1)))
        if p_rvec is None or p_tvec is None:
            return None
        if np.isnan(p_rvec).any() or np.isnan(p_tvec).any():
            return None

        p_tvec = np.array([p_tvec[0][0], p_tvec[1][0], p_tvec[2][0]])

        points_2d = np.array(np.copy(corners)).reshape(np.shape(corners)[0]*np.shape(corners)[2], 2)
        imgs = []
        imgs.append(cv2.aruco.drawAxis(rgb.copy(), camera_matrix, dist_coeff, p_rvec, p_tvec, 0.1))
        if debug:
            cb_pattern = (5,6)
            imgs.append(cv2.aruco.drawAxis(rgb.copy(), camera_matrix, dist_coeff, p_rvec, p_tvec, 0.1))
            imgs.append(cv2.aruco.drawDetectedMarkers(rgb.copy(), corners, ids))
            imgs.append(cv2.drawChessboardCorners(rgb.copy(), cb_pattern, points_2d, True))

        if debug:
            print('--------------------------------')
            print('Translation : {0}'.format(p_tvec))
            print('Rotation    : {0}'.format(p_rvec))
            print('Distance from camera: {0} m'.format(np.linalg.norm(p_tvec)))
            
        # append to lists
        p_rvec_all.append(p_rvec)
        p_tvec_all.append(p_tvec)
        imgs_all.append(imgs)
        points_2d_all.append(points_2d)
        c_corner_all.append(c_corners)
        corners_all.append(corners)
        
    p_tvec_median = np.median(p_tvec_all, axis = 0)       
    p_rvec_median = np.median(p_rvec_all, axis = 0)
    corners_median, c_corner_median, points_2d_median = [],[],[] 
    
    # TODO: how can we fix medianing over a dynamic set (number of points changes) of points?
    #points_2d_median = np.median(points_2d_all, axis = 0)
    #c_corner_median = np.median(c_corner_all, axis = 0)
    #corners_median = np.median(corners_all, axis = 0)
    
    
    hom_TF = tvec_rvec_to_homTF(p_tvec_median, p_rvec_median)
    
    return  p_tvec_median, p_rvec_median, hom_TF, corners_median, c_corner_median, points_2d_median, imgs

def draw(img, K, rvecs, tvecs, points_2d, pattern_3d):
    
    _, rVec, tVec = cv2.solvePnP(np.transpose(pattern_3d), points_2d, K, np.zeros((5,1)))
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, np.zeros((5,1)))
    
    corner = tuple(points_2d[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    
    
    img = cv2.drawChessboardCorners(rgb, cb_pattern, points_2d, ret)
    plt.figure(figsize=(20, 20), dpi=80)
    plt.imshow(img)
    
    return img

