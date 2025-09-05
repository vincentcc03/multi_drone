import torch

def hat(v):
    # v: (..., 3)
    O = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    O[..., 0, 1] = -v[..., 2]
    O[..., 0, 2] =  v[..., 1]
    O[..., 1, 0] =  v[..., 2]
    O[..., 1, 2] = -v[..., 0]
    O[..., 2, 0] = -v[..., 1]
    O[..., 2, 1] =  v[..., 0]
    return O

def Omega(w):
    B = w.shape[0]
    O = torch.zeros(B,4,4, dtype=w.dtype, device=w.device)
    O[:,0,1:] = -w
    O[:,1:,0] = w
    O[:,1,2] = w[:,2]; O[:,1,3] = -w[:,1]
    O[:,2,1] = -w[:,2]; O[:,2,3] = w[:,0]
    O[:,3,1] = w[:,1];  O[:,3,2] = -w[:,0]
    return O

def quat_to_rot(q):
    qw, qx, qy, qz = q[:,0], q[:,1], q[:,2], q[:,3]
    R = torch.zeros(q.shape[0],3,3, dtype=q.dtype, device=q.device)
    R[:,0,0] = 1 - 2*(qy**2+qz**2)
    R[:,0,1] = 2*(qx*qy-qz*qw)
    R[:,0,2] = 2*(qx*qz+qy*qw)
    R[:,1,0] = 2*(qx*qy+qz*qw)
    R[:,1,1] = 1 - 2*(qx**2+qz**2)
    R[:,1,2] = 2*(qy*qz-qx*qw)
    R[:,2,0] = 2*(qx*qz-qy*qw)
    R[:,2,1] = 2*(qy*qz+qx*qw)
    R[:,2,2] = 1 - 2*(qx**2+qy**2)
    return R