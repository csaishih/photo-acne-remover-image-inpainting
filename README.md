# Image Inpainting
Image inpainting can remove objects in a photo and replace them with believable textures.  
The research behind the algorithm can be found [here](../master/criminisi_tip2004.pdf)
Visualization of the algorithm  
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/algo_animation "Inpainting visualization")

## Running the script
###### Dependencies
  * [NumPy](http://www.numpy.org/)
  * [OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

###### Basic usage
`python run.py -s [path to source image] -m [path to mask image] -o [path to output image] -r [patch radius]`

###### Example usage
`python run.py -s test_images/test2/source.png -m test_images/test2/mask.png -o test_images/test2/out.png -r 4`

## Results
###### Test 2 source
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/source.png "Test 2 source")

###### Test 2 mask
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/mask.png "Test 2 mask")

###### Test 2 result with r = 4
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test2/out.png "Test 2 result")


###### Test 5 source
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test5/source.png "Test 5 source")

###### Test 5 mask
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test5/mask.png "Test 5 mask")

###### Test 5 result with r = 4
![alt text](https://github.com/g3aishih/image-inpainting/blob/master/test_images/test5/out.png "Test 5 result")
