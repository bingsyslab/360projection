import cv2
import numpy as np

def deg2rad(d):
  return float(d) * np.pi / 180

def xrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def yrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Face:
  def __init__(self, descr, img, pv, xprj, yprj, expand_coef, rotate, yaw=0, pitch=0):
    self.img = img
    self.descr = descr

    (self.ih, self.iw, _) = img.shape
    self.pv = pv
    self.plane_pt = np.copy(pv)
    self.k = np.dot(self.plane_pt, self.pv)
    self.xprj = xprj
    self.yprj = yprj
    self.expand_coef = expand_coef
    self.rotate = rotate
    
    self.yaw = yaw
    self.pitch = pitch
    
    m = np.dot(yrotation(yaw), xrotation(pitch))
    self.pv = np.dot(m, self.pv)
    self.plane_pt = np.dot(m, self.plane_pt)
    
    self.xprj = np.dot(m, self.xprj)
    self.yprj = np.dot(m, self.yprj)
    

  def intersect(self, lv, pr=False):
    """
    lv - line vector
    pv - plane vector
    k - plane constant
    
    plane equation: x \cdot pv = k
    
    finds the vector where dot(lv*t, pv) = k
    """
    d = np.dot(lv, self.pv)

    if d == 0:
      # parallel lines
      self.ivmag2 = np.inf
      return
    t = self.k / d
    
    if t < 0:
      # Each ray should intersect with both
      # a positive face and negative face.
      # We only want the positive face.
      self.ivmag2 = np.inf
      return
    self.iv = lv * t
    self.ivmag2 = np.dot(self.iv, self.iv)

  def get_face_prj(self):
    a = np.array([[0., 0., 0., -np.dot(self.plane_pt, self.xprj)],
                  [0., 0., 0., -np.dot(self.plane_pt, self.yprj)],
                  [0., 0., 0., 1.]])
    a[0, :3] = self.xprj
    a[1, :3] = self.yprj
    return a.T
  
  def get_img_prj(self):
    ec = self.expand_coef
    ih = self.ih * .5
    iw = self.iw * .5
    if not self.rotate:
      return np.array([[iw / ec, 0., iw],
                       [0., -ih / ec, ih]]).T
    else:
      return np.array([[0., iw / ec, iw - 1],
                       [ih / ec, 0., ih]]).T
  
  def get_isect_pixel(self):
    """
      returns the pixel associated with the plane intersecting vector, self.iv
      
      Note that plane faces are 2 by 2 for a sphere of radius 1.
    """
    if self.ivmag2 == np.inf:
      raise
    
    y = int (np.round((.5 - np.dot(self.iv - self.plane_pt, self.yprj) / self.expand_coef * .5) * self.ih))
    if y < 0.: y = 0
    if y >= self.ih: y = self.ih - 1
    
    x = int (np.round((.5 + np.dot(self.iv - self.plane_pt, self.xprj) / self.expand_coef * .5) * self.iw))
    if x < 0.: x = 0
    if x >= self.iw: x = self.iw - 1
    
    if not self.rotate:
      return self.img[y, x]
    else:
      return self.img[x, self.iw - 1 - y]


class Cube:
  def __init__(self, img, expand_coef, offcenter_z, yaw, pitch, pl_type=False, is_stereo=True):
    [self.InitMono, self.InitStereo][is_stereo](img, expand_coef, offcenter_z, yaw, pitch, pl_type)
        
  def InitStereo(self, img, expand_coef, offcenter_z, yaw, pitch, pl_type):
    self.InitMono(img[:, :img.shape[1] / 2], expand_coef, offcenter_z, yaw, pitch, pl_type)
    
  def InitMono(self, img, expand_coef, offcenter_z, yaw, pitch, pl_type):
    (h, w, _) = img.shape
    
    self.offcenter_z = offcenter_z
    b = 0
    ROTATE = 1
    NO_ROTATE = 0
    a = np.array
    
    if pl_type:  # playlist
      self.faces = [
        Face('left', img[:h / 2, w / 3:w * 2 / 3], a([-1., 0., 0.]), a([ 0., 0., 1.]), a([0., 1., 0.]), expand_coef, NO_ROTATE, yaw, pitch),
        Face('front', img[h / 2:, w / 3:w * 2 / 3], a([ 0., 0., 1.]), a([ 1., 0., 0.]), a([0., 1., 0.]), expand_coef, NO_ROTATE, yaw, pitch),
        Face('right', img[:h / 2, :w / 3], a([ 1., 0., 0.]), a([ 0., 0., -1.]), a([0., 1., 0.]), expand_coef, NO_ROTATE, yaw, pitch),
        Face('top', img[:h / 2, w * 2 / 3:], a([ 0., 1., 0.]), a([ 1., 0., 0.]), a([0., 0., -1.]), expand_coef, NO_ROTATE, yaw, pitch),
        Face('back', img[h / 2:, w * 2 / 3:], a([ 0., 0., -1.]), a([-1., 0., 0.]), a([0., 1., 0.]), expand_coef, NO_ROTATE, yaw, pitch),
        Face('bottom', img[h / 2:, :w / 3], a([ 0., -1., 0.]), a([ 1., 0., 0.]), a([0., 0., 1.]), expand_coef, NO_ROTATE, yaw, pitch),
      ]       
    else:  # playlist_rotated_cubemap or playlist_dynamic_streaming 
      self.faces = [
        Face('left', img[b:h / 3 - b, 0:w / 2], a([-1., 0., 0.]), a([ 0., 0., 1.]), a([0., 1., 0.]), expand_coef, ROTATE, yaw, pitch),
        Face('front', img[h / 3 + b:h * 2 / 3 - b, 0:w / 2], a([ 0., 0., 1.]), a([ 1., 0., 0.]), a([0., 1., 0.]), expand_coef, ROTATE, yaw, pitch),
        Face('right', img[h * 2 / 3 + b:h - b, 0:w / 2], a([ 1., 0., 0.]), a([ 0., 0., -1.]), a([0., 1., 0.]), expand_coef, ROTATE, yaw, pitch),
        Face('top', img[b:h / 3 - b, w / 2:], a([ 0., 1., 0.]), a([-1., 0., 0.]), a([0., 0., 1.]), expand_coef, NO_ROTATE, yaw, pitch),
        Face('back', img[h / 3 + b:h * 2 / 3 - b, w / 2:w], a([ 0., 0., -1.]), a([-1., 0., 0.]), a([0., 1., 0.]), expand_coef, NO_ROTATE, yaw, pitch),
        Face('bottom', img[h * 2 / 3 + b:h - b, w / 2:], a([ 0., -1., 0.]), a([-1., 0., 0.]), a([0., 0., -1.]), expand_coef, NO_ROTATE, yaw, pitch),
      ]

    self.img = img
    self.front_face = self.faces[1].pv
    self.face_vecs = np.zeros((3, 6))
    for i, f in enumerate(self.faces):
      self.face_vecs[:, i] = f.pv / f.k

  def render_view(self, theta0, phi0, rendered_image, fov_h, fov_v):
    """
    given yaw and pitch of head orientation, render view with fov_h * fov_v 
    """
    m = np.dot(yrotation(phi0), xrotation(theta0))
    
    (height, width, _) = rendered_image.shape
  
    DI = np.ones((height * width, 3), np.int)
    trans = np.array([[2.*np.tan(fov_h / 2.) / float(width), 0., -np.tan(fov_h / 2.)],
                      [0., -2.*np.tan(fov_v / 2.) / float(height), np.tan(fov_v / 2.)]])
    
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    DI[:, 0] = xx.reshape(height * width)
    DI[:, 1] = yy.reshape(height * width)
  
    v = np.ones((height * width, 3), np.float)
  
    v[:, :2] = np.dot(DI, trans.T)
    v = np.dot(v, m.T)
    v = v / np.linalg.norm(v, ord=2, axis=1)[:, np.newaxis]
    v += self.offcenter_z * self.front_face
      
    t_inv = np.dot(v, self.face_vecs)
    t_inv_mx = np.argmax(t_inv, 1)
    for i, f in enumerate(self.faces): 
      fvecs = (t_inv_mx == i)
      t_inv_i = t_inv[fvecs, i][:, np.newaxis]
      if t_inv_i.shape[0] == 0: continue
      
      pts = np.ones((t_inv_i.shape[0], 4), np.float)     
      pts[:, :3] = v[fvecs, :] / t_inv_i
      pts = np.rint(np.dot(pts, np.dot(f.get_face_prj(), f.get_img_prj()))).astype(np.int)
       
      rendered_image[DI[fvecs, 1], DI[fvecs, 0]] = f.img[pts[:, 1], pts[:, 0]]

  def cube_to_equi(self, equi_image):
    """
    generate an equirectangular image using the (offset) cube
    if it is an offset cube, the generated equirectangular will clearly show 
    that areas where the front cube face is located is more detailed than other areas 
    """
    (height, width, _) = equi_image.shape
    
    DI = np.ones((height * width, 3), np.int)
    
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    DI[:, 0] = xx.reshape(height * width)
    DI[:, 1] = yy.reshape(height * width)
    
    trans = np.array([[2.*np.pi / float(width), 0., -np.pi],
                      [0., -np.pi / float(height), .5 * np.pi]])
    
    phi_theta = np.dot(DI, trans.T)

    c_theta = np.cos(phi_theta[:, 1])
    s_theta = np.sin(phi_theta[:, 1])
    
    c_phi = np.cos(phi_theta[:, 0])
    s_phi = np.sin(phi_theta[:, 0])
    
    zero = np.zeros(width * height, np.float)
    
    rot = np.array([
                    [c_phi, -s_phi * s_theta, s_phi * c_theta],
                    [zero, c_theta, s_theta],
                    [-s_phi, -c_phi * s_theta, c_phi * c_theta]
                   ])
    
    rot = np.transpose(rot, (2, 0, 1))
    
    v = np.dot(rot, np.array([0., 0., 1.]).T)
    v += self.offcenter_z * self.front_face

    t_inv = np.dot(v, self.face_vecs)
    t_inv_mx = np.argmax(t_inv, 1)
    for i, f in enumerate(self.faces): 
      fvecs = (t_inv_mx == i)
      t_inv_i = t_inv[fvecs, i][:, np.newaxis]
      if t_inv_i.shape[0] == 0: continue
      
      pts = np.ones((t_inv_i.shape[0], 4), np.float)     
      pts[:, :3] = v[fvecs, :] / t_inv_i
      pts = np.rint(np.dot(pts, np.dot(f.get_face_prj(), f.get_img_prj()))).astype(np.int)
       
      equi_image[DI[fvecs, 1], DI[fvecs, 0]] = f.img[pts[:, 1], pts[:, 0]]


def offaxis_cube_to_equi_np(img, yaw, pitch, expand_coef, offcenter_z):  
  equi_image = np.zeros((1000, 2000, 3), np.uint8)
  Cube(img, expand_coef, offcenter_z, yaw, pitch).cube_to_equi(equi_image)

  cv2.imwrite('nnnn_equi_image.jpg', equi_image)


def offaxis_cube_to_render_np(theta0, phi0, yaw, pitch, offcenter_z, fov_h, fov_v):
  rendered_image = np.zeros((1000, 1000, 3), np.uint8)
  Cube(img, expand_coef, offcenter_z, yaw, pitch).render_view(
      deg2rad(theta0), deg2rad(phi0), rendered_image, deg2rad(fov_h), deg2rad(fov_v))
  
  cv2.imwrite('rendered_image_%d_%d.bmp' % (theta0, phi0), rendered_image)


if __name__ == '__main__':
  img = cv2.imread('../scene_1/scene00181-oculus.jpg')

  expand_coef = 1.03125
  offcenter_z = -0.7
  
  # assume yaw and pitch of the center of cube's front face are both 0
  # in rad
  yaw = 0
  pitch = 0
  # draw offaxis cube onto equirectangular
  offaxis_cube_to_equi_np(img, yaw, pitch, expand_coef, offcenter_z)

  # field of view
  fov_h = 90
  fov_v = 90

  # viewing angle
  for theta0, phi0 in [(0, 0), (-45, 330)]:
    offaxis_cube_to_render_np(
        theta0, phi0, yaw, pitch, offcenter_z, fov_h, fov_v)

