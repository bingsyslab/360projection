# 360-degree Projection

This repo includes code for rendering 360-degree images/video frames and converting these images among different types of spherical projections.

## Equirectangular Projection

In equirectangular projection, angles on the sphere given by yaw and pitch values(in degrees) are discretized and mapped to pixels on a rectangular image with x = (yaw + 180)/360 * width and y = (90 - pitch)/180 * height. 

The size of the equirectangular image (width, height). The center of the equirectangular image is at: ```<yaw = 0, pitch = 0>```.

**equirectangular.py** contains code for rendering view at a specific angle, vertical and horizontal field of view, and view resolution, given an image in equirectangular projection. It also contains code for converting the equirectangular projection to the standard cubic projection.

## Cubic Projection

In cubic projections, pixels on a sphere are projected onto six faces of a cube. Six cube faces are then arranged on a planar image for image/video encoding. To date, there exists two types of cubic projections: the standard cubic projection and the offset cubic projection. Standard cube can be seen as a special case of offset cube. 

### Standard Cube

In this projection, a cube is constructed around the sphere. Rays are projected outward from the center of the sphere, and each ray intersects with both a location on the spherical surface and a location on a cube face. Pixels on the spherical surface are mapped to corresponding pixels on the cube through the mapping produced by these projected rays. 

### Offset Cube

The offset cubic projection has an orientation. Distortion is performed on the spherical surface so that pixels on the sphere that are close to the oriented area are mapped to wider angles on the cube. More details on the offset cubic projection can be found in our paper: 

<p align="center">
<strong>A Measurement Study of Oculus 360 Degree Video Streaming</strong> <a href=http://www.cs.binghamton.edu/~yaoliu/publications/mmsys17-360video.pdf>[pdf]</a> <br/>
<i>Chao Zhou, Zhenhua Li, and Yao Liu</i><br/>
Proceedings of ACM Multimedia Systems (MMSys) 2017
</p>

**cube.py** contains code for contains code for rendering view at a specific angle, vertical and horizontal field of view, and view resolution, given an image in cubic projection. It also contains code for converting the cubic projection to the equirectanglar projection.

**equi_to_offset_cube.py** is used for generating OffsetCube given an equirectangular image as well as OffsetCube settings including the yaw and pitch of offset cube's orientation, width and height of offset cube faces (in pixels), and the magnitude of offcenter_z (e.g., -0.7). 

# Usage
```
usage: main.py [-h] config_f img_f

positional arguments:
  config_f    path to config file
  img_f       path to input image

optional arguments:
  -h, --help  show this help message and exit
```
config is a json formatted file. A few example config files as well as a `config_template` have been provided in this repo.

For example, to render an equirectangular image, example parameters can be found at `config_equi_render`, then run
```
python main.py config_equi_render equi_image.jpg
```
To convert an equirectangular image to an offset cube image, use example parameters at `config_equi_to_offcube` and run
```
python main.py config_equi_to_offcube equi_image.jpg
```
To render an offset cube image, offset cube parameters (e.g., yaw and pitch of the offset cube, expand coefficient, and offset value) and rendering parameters must be specified in `config_offcube_render`
```
python main.py config_offcube_render offset_image.jpg
```
