import cv2
import numpy as np

from cube import Cube
from cube import Face

def xrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def yrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def deg2rad(d):
  return d*np.pi/180

class OffsetCube(Cube):
  def __init__(self, expand_coef, offcenter_z, yaw, pitch, w, h, equi_img, bmp_fn=None):
    self.expand_coef = expand_coef
    self.offcenter_z = offcenter_z
    self.yaw = yaw # in rad
    self.pitch = pitch # in rad
    
    self.fw = w # face width
    self.fh = h # face height
    
    self.bmp_fn = bmp_fn
    
    d= {}
    d['left']   = (0,        3*np.pi/2)
    d['front']  = (0,        0)
    d['right']  = (0,        np.pi/2)
    d['back']   = (0,        np.pi)
    d['top']    = (np.pi/2,  np.pi)
    d['bottom'] = (-np.pi/2, np.pi)
    self.d = d
    
    self.init_cube(equi_img)
    
  def init_cube(self, equi_img):
    NO_ROTATE = 0
    a = np.array
    self.faces = [    
        Face('left',   project_offset_face_np(self.pitch, self.yaw, self.d['left'],  self.fh, self.fw, self.expand_coef, self.offcenter_z, equi_img), a([-1.,  0.,  0.]), a([ 0., 0.,  1.]), a([0., 1., 0.]), self.expand_coef, NO_ROTATE, self.yaw, self.pitch),
        Face('front',  project_offset_face_np(self.pitch, self.yaw, self.d['front'], self.fh, self.fw, self.expand_coef, self.offcenter_z, equi_img), a([ 0.,  0.,  1.]), a([ 1., 0.,  0.]), a([0., 1., 0.]), self.expand_coef, NO_ROTATE, self.yaw, self.pitch),
        Face('right',  project_offset_face_np(self.pitch, self.yaw, self.d['right'], self.fh, self.fw, self.expand_coef, self.offcenter_z, equi_img), a([ 1.,  0.,  0.]), a([ 0., 0., -1.]), a([0., 1., 0.]), self.expand_coef, NO_ROTATE, self.yaw, self.pitch),
        Face('top',    project_offset_face_np(self.pitch, self.yaw, self.d['top'],   self.fh, self.fw, self.expand_coef, self.offcenter_z, equi_img), a([ 0.,  1.,  0.]), a([-1., 0.,  0.]), a([0., 0., 1.]), self.expand_coef, NO_ROTATE, self.yaw, self.pitch),
        Face('back',   project_offset_face_np(self.pitch, self.yaw, self.d['back'],  self.fh, self.fw, self.expand_coef, self.offcenter_z, equi_img), a([ 0.,  0., -1.]), a([-1., 0.,  0.]), a([0., 1., 0.]), self.expand_coef, NO_ROTATE, self.yaw, self.pitch),
        Face('bottom', project_offset_face_np(self.pitch, self.yaw, self.d['bottom'],self.fh, self.fw, self.expand_coef, self.offcenter_z, equi_img), a([ 0., -1.,  0.]), a([-1., 0.,  0.]), a([0., 0.,-1.]), self.expand_coef, NO_ROTATE, self.yaw, self.pitch),
        ]
        
    self.front_face = self.faces[1].pv
    self.face_vecs = np.zeros((3,6))
    for i, f in enumerate(self.faces):
      self.face_vecs[:, i] = f.pv / f.k


def project_offset_face_np(theta0, phi0, f_info, height, width, expand_coef, offcenter_z, img):
  """
  theta0 is front pitch
  phi0 is front yaw
  both in radiant
  """
  theta1, phi1= f_info
  
  m = np.dot( yrotation( phi0 ), xrotation( theta0 ) )
  n = np.dot( yrotation( phi1 ), xrotation( theta1 ) ) 
  mn = np.dot( m, n )

  (base_height, base_width, _) = img.shape
  
  new_img = np.zeros((height, width, 3), np.uint8)
    
  DI = np.ones((height*width, 3), np.int)
  trans = np.array([[2./float(width)*expand_coef, 0., -expand_coef],
                    [0.,-2./float(height)*expand_coef, expand_coef]])
  
  xx, yy = np.meshgrid(np.arange(width), np.arange(height))
  
  DI[:, 0] = xx.reshape(height*width)
  DI[:, 1] = yy.reshape(height*width)

  v = np.ones((height*width, 3), np.float)

  v[:, :2] = np.dot(DI, trans.T)
  v = np.dot(v, mn.T)
  
  pv = np.dot(np.array([0, 0, 1.]), m.T)
  
  off = offcenter_z * pv

  a = v[:,0]**2 + v[:,1]**2 + v[:,2]**2   
  b = -2 * (v[:,0]*off[0] + v[:,1]*off[1] + v[:,2]*off[2])
  c = np.sum(off*off)-1
  
  t = (-b+np.sqrt(b*b - 4*a*c))/(2*a)
  
  v = v*t[:,None] - off
  
  diag = np.sqrt(v[:,2]**2 + v[:,0]**2)
  theta = np.pi/2 - np.arctan2(v[:,1],diag) 
  phi = np.arctan2(v[:,0],v[:,2]) + np.pi

  ey = np.rint(theta*base_height/np.pi ).astype(np.int) 
  ex = np.rint(phi*base_width/(2*np.pi) ).astype(np.int) 
  
  ex[ex >= base_width] = base_width - 1
  ey[ey >= base_height] = base_height - 1  

  new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]
    
  return new_img


if __name__== '__main__':
  image = cv2.imread('../youtube_equi_image.jpg')

  cube_yaw = 0
  cube_pitch = 0

  off_cb = OffsetCube(1.03125, -0.7, deg2rad(cube_yaw), deg2rad(cube_pitch), 528, 528, image)
  
  for face in off_cb.faces:
    cv2.imwrite('generated_offset_cube_%s.bmp'%face.descr, face.img)
