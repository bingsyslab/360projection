from string import Template

orientation = '''"orientation" : {
        "yaw" : "in degrees",
        "pitch" : "in degrees"
      }'''

projection = Template('''{
    "resolution" : {
      "width" : "in pixels",
      "height" : "in pixels"
    },
    "equi" : {},
    "cube" : {
      $orientation,
      "exp_coef" : "a float between 1 and 1.05 indicating the percentage of extra pixels to encode on a cube face",
      "offset" : "between 0 and -1, the offset in the orientation of the offset cube to apply to vectors pointing to orientations on the sphere"
    },
    "render" : {
      "fov" : {
        "horizontal" : "horizontal FOV in degrees",
        "vertical" : "vertical FOV in degrees"
      },
      $orientation
    }
  }''').substitute({'orientation' : orientation})

config = Template('''
{
  "input" : $projection,
  "output" : $projection
}
''').substitute({'projection' : projection})

print config
