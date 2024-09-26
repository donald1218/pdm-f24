import cv2
import numpy as np
import math

points = []


class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        self.points = points


    def top_to_front(self,theta=0, phi=0, gamma=0, dx=0, dy=-1.5, dz=0, fov=90):

        # Focal length based on FOV 
        f = self.width / (2.0 * np.tan(np.radians(fov) / 2.0))  
        # width = height => fw = fh

        # Linear transformation matrix from C2 to C1
        # T^C1_C2 = T^C1_W * T^W_C2
        R_t_C2ToW = self.get_Trans_w_c(np.deg2rad(theta),0,0, 0, -2.5, 0)
        R_t_C1ToW = self.get_Trans_w_c(0,0,0, 0, -1, 0)
        R_t_C2ToC1 = np.dot(np.linalg.inv(R_t_C1ToW),R_t_C2ToW)
        
        # Intrinsic camera matrix
        K = np.array([[f, 0,self.width/2],
            [0, f, self.height/2],
            [0, 0, 1]])


        new_pixels = []
       
        for i in range(len(self.points)):
            # assumption: depth is 2.5 
            depth = 2.5
            # convert [u,v] to [x,y,z] by multiplying depth and inverse of intrinsic matrix
            image_coor = np.dot(np.linalg.inv(K),np.array([[depth*self.points[i][0]],[depth*self.points[i][1]],[depth]]))
            # convert [x,y,z] to [X,Y,Z,1] by adding 1 and multiply with transformation matrix
            p = np.dot(R_t_C2ToC1, np.array([image_coor[0][0],image_coor[1][0],image_coor[2][0],1]))
            # convert [X,Y,Z,1] to [x,y,z] by K*[I|O]*[X,Y,Z,1]
            d = np.dot(K, np.array([p[0],p[1],p[2]]))
            # convert [x,y,z] to [u,v] by dividing depth
            new_pixels.append([np.round(d[0]/d[2]),np.round(d[1]/d[2])])
        return new_pixels
    
    def get_Trans_w_c(self,theta, phi, gamma, dx, dy, dz):
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)]])

        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
        
        R = np.dot(np.dot(Rz,Ry),Rx)
        
        T = np.array([dx,dy,dz])
        
        R_t = np.eye(4)
        
        R_t[0:3,0:3] = R
        
        R_t[0:3,3] = T
        

        return R_t
        


    def show_image(self, new_pixels, img_name, color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """
        print(new_pixels)
        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels,dtype="int32")], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"
    front_rgb_2 = "bev_data/front2.png"
    top_rgb_2 = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels,'projection1.png')
    
    points.clear()
    img = cv2.imread(top_rgb_2, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    projection = Projection(front_rgb_2, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels,'projection2.png')
