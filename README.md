# Image Inpainting

###### Dependencies
..*[NumPy](http://www.numpy.org/)
..*[OpenCV][http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html]

###### Basic usage
`python run.py -s [path to source image] -m [path to mask image] -o [path to output image] -r [patch radius]`

###### Example usage
`python run.py -s test_images/test1/source.png -m test_images/test1/mask.png -o test_images/test1/out.png -r 4`

# Results
###### Source
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/source.png "Test2 source")

###### Result with r = 4
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/out.png "Test2 result")
