# Image Inpainting

###### Dependencies
  * [NumPy](http://www.numpy.org/)
  * [OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

###### Basic usage
`python run.py -s [path to source image] -m [path to mask image] -o [path to output image] -r [patch radius]`

###### Example usage
`python run.py -s test_images/test2/source.png -m test_images/test2/mask.png -o test_images/test2/out.png -r 4`

# Results
###### Test 2 source
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/source.png "Test 2 source")

###### Test 2 mask
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/mask.png "Test 2 mask")

###### Test 2 result with r = 4
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/out.png "Test 2 result")
