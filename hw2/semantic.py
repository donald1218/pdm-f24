import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import cv2
import scipy.io
import pandas as pd
from scipy.spatial import cKDTree


start_point = []

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print("start point",x, ' ', y)
        start_point.append(x)
        start_point.append(y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(map, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('map', map)
        


def RRT(start, goal, step_length,map,end_distance):
    X_set = [start]
    X_last = [start]
    while True:  
        img = cv2.imread('map.png')
        random_p = [np.random.randint(0, img.shape[1] - 1),np.random.randint(0, img.shape[0] - 1)]
        nearest_p = X_set[0]
        # print(random_p)
        for i in range (len(X_set)):
            if distance(X_set[i], random_p) < distance(nearest_p, random_p):
                nearest_p = X_set[i]
        points = bresenham(random_p,nearest_p)
        
        if  not obstacle_check(points, img):
            for i in range (len(points)):
                if distance(nearest_p, points[i]) > step_length:
                    print("add point",points[i-1])
                    X_set.append(points[i-1])
                    X_last.append(nearest_p)
                    cv2.line(map,nearest_p,points[i-1],(0,0,255),1)
                    break
        
        if distance(X_set[len(X_set)-1], goal) < end_distance:
            break
        
    for point in X_set:
        cv2.circle(map, point, radius=1, color=(0, 0, 255), thickness=-1)  # Red color
        
    path = []
    x_now = X_set[len(X_set)-1]
    path.append(x_now)
    while ((x_now[0] != start[0]) and (x_now[1] != start[1])) :
        cv2.line(map,x_now,X_last[X_set.index(x_now)],(0,255,0),2)
        x_now = X_last[X_set.index(x_now)] 
        path.append(x_now)
        
    path.reverse()
        
        
    cv2.imshow('Map with Path', map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return path

def obstacle_check(points, map):
    for i in range(len(points)):
        if map[points[i][1]][points[i][0]][0] != 255 or map[points[i][1]][points[i][0]][1] != 255 or map[points[i][1]][points[i][0]][2] != 255:
            return True
        if points[i][0] > 226 and points[i][0] < 244 and points[i][1] > 91 and points[i][1] < 96:
            return True
    return False

def bresenham(random_p,nearest_p):
    points = []
    x1, y1 = random_p
    x2, y2 = nearest_p
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    sx = 1 if x2 < x1 else -1
    sy = 1 if y2 < y1 else -1
    err = dx - dy

    while True:
        points.append((x2, y2))
        if x2 == x1 and y2 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x2 += sx
        if e2 < dx:
            err += dx
            y2 += sy
            
    return points


def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

if __name__ == '__main__':
    points = np.load("semantic_3d_pointcloud/point.npy") *10000/255
    colors = np.load("semantic_3d_pointcloud/color01.npy")
    colors255 = np.load("semantic_3d_pointcloud/color0255.npy")
    unremoved_p = ~((points[:,1] < -1.2) |(points[:,1] > 0))
    colors255_r = colors255[unremoved_p]
    
    plt.figure()
    plt.scatter(points[unremoved_p, 2], points[unremoved_p, 0], s=1.5, c=colors[unremoved_p])
    plt.axis('off')
    plt.savefig('map.png', dpi=144, bbox_inches='tight', pad_inches=0)
    
    
       # before running the program, make sure to pip install openpyxl
    df = pd.read_excel("color_coding_semantic_segmentation_classes.xlsx",usecols=["Color_Code (R,G,B)", "Name"])
    
    target_object = ""
    color_code = ""
    
    while True:
        target_object = input('target : ')
        if target_object in df["Name"].values:
            color_code = df["Color_Code (R,G,B)"][df["Name"] == target_object].values[0]
            break
        else:
            print("Invalid object")
            
    print(color_code)
    map = cv2.imread("map.png")
    
    target_points = []
    path_points = []
    
    c = [float(x) if '.' in x else int(x) for x in color_code.strip('()').split(',')]
    
    for i in range(len(map)):  
        for j in range(len(map[0])):
            if map[i][j][0] == c[2] and map[i][j][1] == c[1] and map[i][j][2] == c[0]:
                target_points.append([j, i])
            if map[i][j][0] == 255 and map[i][j][1] == 255 and map[i][j][2] == 255:
                path_points.append([j, i])
                
                
    cv2.imshow('map', map)
    cv2.setMouseCallback('map', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    x_sum, x_max, x_min = 0, 0, 1000
    y_sum, y_max, y_min = 0, 0, 1000
    for i in range(len(target_points)):
        x_max = max(x_max, target_points[i][0])
        x_min = min(x_min, target_points[i][0])
        y_max = max(y_max, target_points[i][1])
        y_min = min(y_min, target_points[i][1])
        x_sum += target_points[i][0]
        y_sum += target_points[i][1]
        
    x_mean = int(x_sum/len(target_points))
    y_mean = int(y_sum/len(target_points))
    
    target = [x_mean, y_mean]
    print("target",target)
    end_distance = np.sqrt((x_max-x_min)**2 + (y_max-y_min)**2)
    
    path = RRT(start_point, target, 20, map,min(end_distance,60))
    
    
    
    
   