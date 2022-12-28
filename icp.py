import copy
import numpy as np
import numpy.linalg as LA
import open3d as o3d
# import pdb; pdb.set_trace()

# 点群のインポート
pcd1 = o3d.io.read_point_cloud("bunny001.pcd")
pcd2 = o3d.io.read_point_cloud("bunny002.pcd")

# ダウンサンプリング
pcd_s = pcd1.voxel_down_sample(voxel_size=0.003)
pcd_t = pcd2.voxel_down_sample(voxel_size=0.003)

# 変換前のソース点群を複製
pcd_s1 = pcd_s

pcd_t.paint_uniform_color([1.0,0.0,0.0]) # ターゲット点群を赤で表示
pcd_s1.paint_uniform_color([0.0,1.0,0.0]) # 変換前のソース点群を緑で表示
o3d.visualization.draw_geometries([pcd_t, pcd_s1])
#ポイント数
n_points = len(pcd_s.points)

# パラメータの設定
iterations = 1000
th_distance = 0.0005
th_ratio = 1.00
closest_indices = []
pcds = []
final_trans = np.identity(4)
d = []

# 回展行列に単位ベクトルを代入（初期化）
q = np.array([1.,0.,0.,0.,0.,0.,0.])
# 最近傍点探索のためのkd-tree構築
pcd_tree = o3d.geometry.KDTreeFlann(pcd_t)
# ターゲットの点群をnumpy配列に変換
np_pcd_t = np.asarray(pcd_t.points)

def closest_points():
    idx_list = []
    distance = []
    for i in range(n_points):
        [k, idx, d] = pcd_tree.search_knn_vector_3d(pcd_s.points[i], 1)
        idx_list.append(idx[0])
        distance.append(d[0])

    np_pcd_y = np_pcd_t[idx_list]
    closest_indices.append( idx_list )
    d.append(np.sqrt(np.mean(np.array(distance))))

    return np_pcd_y.copy()

def compute_registration_param(np_pcd_y):
    # 重心の取得
    mu_s = pcd_s.get_center()
    mu_y = np.mean(np_pcd_y, axis=0)
    # 共分散行列の計算
    np_pcd_s = np.asarray(pcd_s.points)
    covar = np.zeros( (3,3) )
    n_points = np_pcd_s.shape[0]
    for i in range(n_points):
        covar += np.dot( np_pcd_s[i].reshape(-1, 1), np_pcd_y[i].reshape(1, -1) )
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
            pcd_s.transform(transform)
            final_trans = np.dot(transform,final_trans)
            pcds.append(copy.deepcopy(pcd_s))
            # Step 4.
            if (dis[-1] < 0):
                dis[-1] = dis[-1] * -1
            
            if ((2<i) and (th_ratio < dis[-1]/dis[-2])) or (dis[-1] < th_distance):
                break

print("# of iterations: ", i)
print("Registration error [m/pts.]:", dis[-1] )
print("Final transformation \n", final_trans )

pcd_s2 = pcd_s
pcd_s2.paint_uniform_color([0.0,0.0,1.0]) # 変換後のソース点群を青で表示
o3d.visualization.draw_geometries([pcd_t, pcd_s1, pcd_s2])

