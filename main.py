import sys
import json
import argparse

import cv2
import numpy as np

def deg2rad(d):
  return float(d) * np.pi / 180

import cube as c
import equi_to_offset_cube as e2c
import equirectangular as e
      
if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument('config_f', help = 'path to config file')
  parser.add_argument('img_f', help = 'path to input image')

  args = parser.parse_args(sys.argv[1:])
  
  img_f = args.img_f
  img = cv2.imread(img_f)
  
  config_f = args.config_f
  with open(config_f) as f:
    j = json.load(f)
  
  try:
    assert(len(j[u'input']) == 1)
  except AssertionError:
    print "more than one input projection types"
    
  if u'equi' in j[u'input']:
    if u'render' in j[u'output']: # render
      rimg = e.render_image_np(deg2rad(j[u'output'][u'render'][u'orientation'][u'pitch']), \
                               deg2rad(j[u'output'][u'render'][u'orientation'][u'yaw']), \
                               deg2rad(j[u'output'][u'render'][u'fov'][u'vertical']), \
                               deg2rad(j[u'output'][u'render'][u'fov'][u'horizontal']), \
                               j[u'output'][u'resolution'][u'width'], \
                               img
                               )
      
      cv2.imwrite('%s_%d_%d.bmp' % (img_f.split('/')[-1].split('.')[0], \
                                    j[u'output'][u'render'][u'orientation'][u'yaw'], \
                                    j[u'output'][u'render'][u'orientation'][u'pitch']), \
                                    rimg
                                    )
      
    elif u'cube' in j[u'output']:
      face_size = j[u'output'][u'resolution'][u'width'] / 3
      
      cb = e2c.OffsetCube(j[u'output'][u'cube'][u'expand_coef'], \
                          j[u'output'][u'cube'][u'offset'], \
                          deg2rad(j[u'output'][u'cube'][u'orientation'][u'yaw']), \
                          deg2rad(j[u'output'][u'cube'][u'orientation'][u'pitch']), \
                          face_size, \
                          face_size, \
                          img
                          )

      if abs(j[u'output'][u'cube'][u'offset']) < 0.0001:
        output_name = '%s_cube.bmp' %img_f.split('/')[-1].split('.')[0]
      else:
        output_name = '%s_offcube.bmp' %img_f.split('/')[-1].split('.')[0]

      e2c.write_to_cb_img(cb, face_size, output_name)
      
  elif u'cube' in j[u'input']:
    cb = c.Cube(img, \
                j[u'input'][u'cube'][u'expand_coef'], \
                j[u'input'][u'cube'][u'offset'], \
                deg2rad(j[u'input'][u'cube'][u'orientation'][u'yaw']), \
                deg2rad(j[u'input'][u'cube'][u'orientation'][u'pitch']), \
                is_stereo=False)
    
    if u'render' in j[u'output']: # render
      rendered_width = j[u'output'][u'resolution'][u'width']
      rendered_height = j[u'output'][u'resolution'][u'height']
      rendered_image = np.zeros((rendered_height, rendered_width, 3), np.uint8)

      cb.render_view(deg2rad(j[u'output'][u'render'][u'orientation'][u'pitch']), \
                     deg2rad(j[u'output'][u'render'][u'orientation'][u'yaw']), \
                     rendered_image, \
                     deg2rad(j[u'output'][u'render'][u'fov'][u'horizontal']), \
                     deg2rad(j[u'output'][u'render'][u'fov'][u'vertical'])
                     )

      cv2.imwrite('%s_%d_%d.bmp' % (img_f.split('/')[-1].split('.')[0], \
                                    j[u'output'][u'render'][u'orientation'][u'yaw'], \
                                    j[u'output'][u'render'][u'orientation'][u'pitch']), \
                                    rendered_image
                                    )
        
    elif u'equi' in j[u'output']:
      equi_width  = j[u'output'][u'resolution'][u'width']
      equi_height = j[u'output'][u'resolution'][u'height']
      equi_image = np.zeros((equi_height, equi_width, 3), np.uint8)
      
      cb.cube_to_equi(equi_image)
    
      cv2.imwrite('%s_equi.bmp'%img_f.split('/')[-1].split('.')[0], equi_image)
