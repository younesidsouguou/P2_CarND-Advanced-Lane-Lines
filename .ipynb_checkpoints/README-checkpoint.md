
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### Writeup / README

In order to build this project, I took the following steps

## 1. Camera Calibration

I used many images of a chessoard taken from different angles and by finding their corners and ampping them to an absolute grid, we came to build a calibration matrix along with the distortion coefficients. Using these, we become able to estimate the distortion of our pinhole camera, and finally apply a distortion correction to next images.

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./output_images/distorted.png "Distorted"
![alt text][image1]
![alt text][image2]

## 2. Apply a distortion correction to test images.

So now that we calculated our distortion coefficients, we are going to undistort our test image.

[image3]: ./output_images/distorted_test.png "Distorted"
[image4]: ./output_images/undistorted_test.png "Undistorted"

![alt text][image4]
![alt text][image3]

## 3. Use color transforms, gradients, etc., to create a thresholded binary image.

So this is the tricky part in the whole project. Since, it is provinding some input or intelligence to our algorithm, whether it can detect lanes and only lanes or it's easily biased by shadows and light variations. So here we tried to combine a bunch of all the filters we learned from the classroom. We applied a magnitude filter , a gradient filter on the x axis, a directon gradient filter and finally a thresholding on both saturation channel (HLS) and red channel (RGB).

So you can find below the output binary:

[image5]: ./output_images/thresholding.png "Thresholded"

![alt text][image5]

## 4. Apply a perspective transform to rectify binary image ("birds-eye view").

In this part I made a perspective transform. So we define our transform using a source polygone and another destination polygone. In order to find the right coordinates for the source/destination polygones, I used the straight lanes image, and I kept tuning my transformation so that the birds-eye view corresponds a straight parallel lanes. 

This is the original undistorted image versus the warped image (birds-eye view):

P.S: The provided warped image is cropped in order to keep only the region of interest for the next coming steps

[image6]: ./output_images/straight_lines1.jpg
[image7]: ./output_images/warped_rgb.png
![alt text][image6]
![alt text][image7]

## 5. Detect lane pixels and fit to find the lane boundary.

Once with the warped image, we immediately apply the binary transform on it. 

[image8]: ./output_images/warped_binary.png
![alt text][image8]

We crop our region of interest (lanes region)

[image9]: ./output_images/cropped_warped_binary.png
![alt text][image9]

At this moment we sum on the y-axis in order to get our historgram pixels intesity in x-axis and therefore the starting position of our sliding windows

[image10]: ./output_images/x_sum_histogram.png
![alt text][image10]

Now it's time to put up some sliding windows and keep searching for the areas with many pixels. In order to make it easy for helper function `find_lane_pixels` in `example.py` file, I added the following paramaters:

`nwindows` : number of sliding windows<br>
`margin` : the width of the sliding window<br>
`perc_pixels` : the minimum percentage of the white pixels in the window to consider an update a lane position update <br>

I finally go into some polynomial fitting of the averaged positions given by the sliding windows. I get the coefficients for both left and right lanes. Finally, I make a curve array of coordinates in order to trace my lanes fill the area between lanes (in the bird-eyes view)

[image11]: ./output_images/lanes_fit.png
![alt text][image11]

## 6. Determine the curvature of the lane and vehicle position with respect to center.

Now it's the easy part, we have the quadradtic coefficients of each lane from the previous parts using this formula:

[image12]: ./output_images/roc.png
![alt text][image12]


And right after I applied the transformation from pixel to real-world.

For the position of the vehicule I took the ratio between `(Center position - Left_lane_position)` and `(Right_lane_position - Left lane position)` then I substracted `0.5` and multiplied it by the `lane_width`. So a postive value indicates an offset the right and a negative value is an offset to the left.

## 7. Warp the detected lane boundaries back onto the original image.

It's time to put the detections on our original undistorted image. So I warped back the detected lanes by inversing the perspective transform of the detected lanes and road.

[image13]: ./output_images/unwarped_lanes_fit.png
![alt text][image13]


And right after I added it to the original image.

[image14]: ./output_images/final_img.png
![alt text][image14]

## 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Finally, we're puuting some text (Radius of curves and Vehicle position) onto our final frame.

[image15]: ./output_images/text.png
![alt text][image15]
