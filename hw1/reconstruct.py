import numpy as np
import open3d as o3d
import argparse
import os
from scipy.spatial.transform import Rotation
import cv2


def depth_image_to_point_cloud(rgb, depth):
    # TODO: Get point cloud from rgb and depth image
     
    # from spec
    width, height = rgb.shape[0],rgb.shape[1]
    fov, depth_scale = 90, 1000
    f = width / (2.0 * np.tan(np.radians(fov) / 2.0)) 
    
    v,u = np.mgrid[0:width,0:height]
    
    z = depth[:,:,0] * depth_scale
    z = depth[:, :, 0].astype(np.float32) / 255 * (-10)
    x = (u - width/2) * z / f
    y = (v - height/2) * z / f
    
    points = np.stack((x,y,z),axis=-1).reshape(-1,3)
    
    # rgbs = rgb.reshape(-1,3)
    # RGB to BRG and normalize
    rgbs = (rgb[:, :, [2, 1, 0]].astype(np.float32) / 255).reshape(-1, 3)
    
    # create PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # filter out points with depth > 1.5m
    depth_threshold = 80 / 255 * 10
    filt = points[:, 2] <= depth_threshold
    points = points[filt,:]
    rgbs = rgbs[filt,:]
    
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



def execute_global_registration(source_down, target_down, source_fpfh,target_fpfh, voxel_size):
    
    distance_threshold = voxel_size * 1.5

    checker = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, checker, o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
    
    return result



def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    
    result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
    
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    raise NotImplementedError
    return result


def reconstruct(args):
    # TODO: Return results
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    points_clouds = get_clouds(args)
    
    pcd_downs,pcd_fpfhs = [],[]
    num = len(points_clouds)
    for i in range (num):
        pcd_down,pcd_fpfh = preprocess_point_cloud(points_clouds[i], voxel_size=0.07)
        pcd_downs.append(pcd_down)
        pcd_fpfhs.append(pcd_fpfh)
        
    pred_cam_pos = [np.eye(4)]
    
    for i in range (1,num):
        d_source,d_target = pcd_downs[i],pcd_downs[i-1]
        f_source,f_target = pcd_fpfhs[i],pcd_fpfhs[i-1]
        trans_init = execute_global_registration(d_source, d_target, f_source, f_target, voxel_size=0.07).transformation
        trans = trans_init
        if args.version == 'open3d':
            trans = local_icp_algorithm(d_source, d_target,trans_init, threshold=0.07*0.4).transformation
        elif args.version == 'my_icp':
            trans = my_local_icp_algorithm(d_source, d_target,trans_init, threshold=0.05)
        pred_cam_pos.append(pred_cam_pos[i-1]@trans)
    
    for i in range (num):
        points_clouds[i].transform(pred_cam_pos[i])
            
    return points_clouds, np.array(pred_cam_pos)

def get_clouds(args):
    point_clouds = []
    num = len(os.listdir(os.path.join(args.data_root, "depth")))
    
    for i in range (num):
        rgb = cv2.imread(args.data_root + "/rgb/" + str(i+1) + ".png")
        depth = cv2.imread(args.data_root + "/depth/" + str(i+1) + ".png")
        pcd = depth_image_to_point_cloud(rgb, depth)
        point_clouds.append(pcd)
        
    return point_clouds
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_cam_pos = reconstruct(args)
    
    for i in range(len(result_pcd)):
        points = np.asarray(result_pcd[i].points)
        rgbs = np.asarray(result_pcd[i].colors)
        valid = (points[:, 1] <= 0.25)
        points, rgbs = points[valid], rgbs[valid]

        result_pcd[i].points = o3d.utility.Vector3dVector(points[:, 0:3])
        result_pcd[i].colors = o3d.utility.Vector3dVector(rgbs[:, 0:3])
    
    # x y z /x y z w
    ground_truth_pos = np.load(args.data_root + "/GT_pose.npy")
    gtc_pos = np.tile(np.eye(4),(ground_truth_pos.shape[0],1,1))
    
    # convert quaternion to rotation matrix
    gtc_pos[:,0:3,0:3] = Rotation.from_quat(ground_truth_pos[:,3:7]).as_matrix()
    # gtc_pos[:,0:3,0:3] = euler_angle.as_matrix()
    # set translation
    gtc_pos[:,0:3,3] = ground_truth_pos[:,0:3]
    # base on 1.png gtc_pos to change coordinate 
    gtc_pose = np.tile(np.linalg.inv(gtc_pos[[0]]), (gtc_pos.shape[0], 1, 1)) @ gtc_pos
    gt_cam_position = gtc_pose[:, 0:3, 3]

    gt_cam_position[:,2] = -gt_cam_position[:,2]
    gt_cam_position[:,0] = -gt_cam_position[:,0]
    pred_cam_position = pred_cam_pos[:, 0:3, 3]

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    print("Mean L2 distance: ", np.mean(np.linalg.norm(gtc_pose - pred_cam_pos)))

    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    edges = [[i, i+1] for i in range(len(gtc_pos) - 1)]
    black_line = [[0, 0, 0] for i in range(len(edges))]
    red_line = [[1, 0, 0] for i in range(len(edges))]
    
    black_line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(gt_cam_position),
        lines = o3d.utility.Vector2iVector(edges)
        )
    
    black_line_set.colors = o3d.utility.Vector3dVector(black_line)

    red_line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(pred_cam_position),
        lines = o3d.utility.Vector2iVector(edges)
        )
    red_line_set.colors = o3d.utility.Vector3dVector(red_line)


    o3d.visualization.draw_geometries(result_pcd+[black_line_set, red_line_set])
    

