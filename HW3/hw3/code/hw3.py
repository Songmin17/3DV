import numpy as np 
import cv2


def find_bbox(keypoints): 
    # bbox algorithm: min, max of x, y
    np_kpts = np.asarray(keypoints)
    min_x, max_x = np.min(np_kpts[:, 0]), np.max(np_kpts[:, 0])
    min_y, max_y = np.min(np_kpts[:, 1]), np.max(np_kpts[:, 1])

    min_x, max_x, min_y, max_y = int(min_x), int(max_x), int(min_y), int(max_y) 
    
    # make corners in clockwise order
    bbox = np.asarray([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    return bbox

def drawKeypoints(img, keypoints, color):
    for i, elem in enumerate(keypoints):
        x, y = elem[0], elem[1]
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 1, color, -1)

    return img 

def drawSkeleton(img, joints, keypointColor, lineColor):
    edges = np.array([[0, 1], [0, 2], [3, 0], [1, 4],
                      [2, 5], [6, 3], [4, 7], [5, 8],
                      [9, 6], [7, 10], [8, 11], [12, 9],
                      [13, 9], [14, 9], [15, 12], [16, 13],
                      [17, 14], [18, 20], [19, 21], [20, 22],
                      [21, 23], [24, 15], [17, 19], [16, 18]])
    
    for edge in edges:
        joint_coord = joints[edge]
        coord1, coord2 = tuple(map(int, joint_coord[0])), tuple(map(int, joint_coord[1]))
        img = cv2.circle(img, coord1, 1, keypointColor, -1)
        img = cv2.circle(img, coord2, 1, keypointColor, -1)
        img = cv2.line(img, coord1, coord2, lineColor, 1)
    
    return img 

def exportMesh(savepath, vertices, faces):
    # write all vertices
    # add 1 to all face indices, write each face
    with open(savepath, 'w') as f:
        for vertex in vertices:
            v1, v2, v3 = vertex[0], vertex[1], vertex[2]
            f.write(f'v {v1} {v2} {v3}\n')
        for face in faces:
            f1, f2, f3 = face[0]+1, face[1]+1, face[2]+1
            f.write(f'f {f1} {f2} {f3}\n')


# helper function for drawMesh. 
# Projects the 3d vertices to the image plane
def projectPoints(vertices, f, cam_c, cam_rot, cam_t):
    center = cam_c.copy() 
    center /= 2
    t = cam_t.copy()
    t[-1] *= 2
    
    mat1, mat2 = np.zeros((3, 3)), np.zeros((3, 4))
    mat1[0, 0], mat1[1, 1], mat1[:2, 2], mat1[2, 2] = f[0], f[1], center, 1
    mat2[0:3, 0:3], mat2[:, 3] = np.eye(3, 3), t
    h_vertices = np.column_stack((vertices, np.ones(vertices.shape[0])))
    p_vertices = mat1 @ mat2 @ h_vertices.T
    p_vertices = p_vertices.T 
    p_vertices[:, 0] /= p_vertices[:, 2]
    p_vertices[:, 1] /= p_vertices[:, 2]
    return (p_vertices[:, :2]).astype(int)

def drawMesh(img, vertices, faces, lineColor): 
    # params['verts'], params['faces']
    # assume: 3d coords
    for face_ind in faces:
        face_pts = vertices[face_ind]
        coord1, coord2, coord3 = tuple(map(int, face_pts[0])), tuple(map(int, face_pts[1])), tuple(map(int, face_pts[2]))
        img = cv2.line(img, coord1, coord2, lineColor, 1)
        img = cv2.line(img, coord2, coord3, lineColor, 1)
        img = cv2.line(img, coord1, coord3, lineColor, 1)

    return img