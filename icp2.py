"""
''''''''''''''''''''''''''''''''''''''''
Operating Environment:
Ubuntu 22.04 lts 
python 3.10.6
References
"詳解3次元点群処理" by 金崎 朝子, 秋月 秀一, 千葉 直也 (2022), 講談社.

''''''''''''''''''''''''''''''''''''''''
"""

import copy
import numpy as np
import numpy.linalg as LA
import open3d as o3d
# import pdb; pdb.set_trace()

# 点群のインポート
pcd1 = o3d.io.read_point_cloud("bunny001.pcd")
pcd2 = o3d.io.read_point_cloud("bunny002.pcd")

# ダウンサンプリング
point_soruce = pcd1.voxel_down_sample(voxel_size=0.003)
point_target = pcd2.voxel_down_sample(voxel_size=0.003)

# 変換前のソース点群を複製
point_soruce1 = point_soruce

point_target.paint_uniform_color([1.0,0.0,0.0]) # ターゲット点群を赤で表示
point_soruce1.paint_uniform_color([0.0,1.0,0.0]) # 変換前のソース点群を緑で表示
o3d.visualization.draw_geometries([point_target, point_soruce1])
#ポイント数
n_points = len(point_soruce.points)

# パラメータの設定
iterations = 1000
threshold_distance = 0.0007
threshold_error = 0.999
closest_indices = []
pcds = []
final_transform = np.identity(4)
d = []

# 回展行列に単位ベクトルを代入（初期化）
q = np.array([1.,0.,0.,0.,0.,0.,0.])

# 最近傍点探索のためのkd-tree構築
point_targetree = o3d.geometry.KDTreeFlann(point_target)

# ターゲットの点群をnumpy配列に変換
np_point_target = np.asarray(point_target.points)

def closest_points():
    idx_list = []
    distance = []
    for i in range(n_points):
        [k, idx, d] = point_targetree.search_knn_vector_3d(point_soruce.points[i], 1)
        idx_list.append(idx[0])
        distance.append(d[0])

    np_pcd_y = np_point_target[idx_list]
    closest_indices.append( idx_list )
    d.append(np.sqrt(np.mean(np.array(distance))))

    return np_pcd_y.copy()

def compute_registration_param(np_pcd_y):
    # 重心の取得
    mu_s = point_soruce.get_center()
    mu_y = np.mean(np_pcd_y, axis=0)
    # 共分散行列の計算
    np_point_soruce = np.asarray(point_soruce.points)
    covar = np.zeros( (3,3) )
    n_points = np_point_soruce.shape[0]
    for i in range(n_points):
        covar += np.dot( np_point_soruce[i].reshape(-1, 1), np_pcd_y[i].reshape(1, -1) )
    covar /= n_points
    covar -= np.dot( mu_s.reshape(-1,1), mu_y.reshape(1,-1) )
    ## anti-symmetrix matrix
    A = covar - covar.T
    delta = np.array([A[1,2],A[2,0],A[0,1]])
    tr_covar = np.trace(covar)
    i3d = np.identity(3)
    # symmetric matrixの作成
    Q = np.zeros((4,4))
    Q[0,0] = tr_covar
    Q[0,1:4] = delta
    Q[1:4,0] = delta
    Q[1:4,1:4] = covar + covar.T - tr_covar*i3d
    w, v = LA.eig(Q)
    rot = quaternion2rotation(v[:,np.argmax(w)])
    trans = mu_y - np.dot(rot,mu_s)
    q = np.concatenate((v[np.argmax(w)],trans))
    transform = np.identity(4)
    transform[0:3,0:3] = rot.copy()
    transform[0:3,3] = trans.copy()
    return transform

def quaternion2rotation(q):
    rot = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
                    [2.0*(q[1]*q[2]+q[0]*q[3]),q[0]**2+q[2]**2-q[1]**2-q[3]**2,2.0*(q[2]*q[3]-q[0]*q[1])],
                    [2.0*(q[1]*q[3]-q[0]*q[2]),2.0*(q[2]*q[3]+q[0]*q[1]),q[0]**2+q[3]**2-q[1]**2-q[2]**2]])
    return rot
    
for i in range(iterations):
            # Step 1.
            np_pcd_y = closest_points()
            dis = np_pcd_y[1]
            # Step 2.
            transform = compute_registration_param(np_pcd_y)
            # Step 3.
            point_soruce.transform(transform)
            final_transform = np.dot(transform,final_transform)
            pcds.append(copy.deepcopy(point_soruce))
            # Step 4.
            if (dis[-1] < 0):
                dis[-1] = dis[-1] * -1
            
            if ((2<i) and (threshold_error < dis[-1]/dis[-2])) or (dis[-1] < threshold_distance):
                break

print("iterations: ", i)
print("Final transformation \n", final_transform )

point_soruce2 = point_soruce
point_soruce2.paint_uniform_color([0.0,0.0,1.0]) # 変換後のソース点群を青で表示
o3d.visualization.draw_geometries([point_target, point_soruce2])

