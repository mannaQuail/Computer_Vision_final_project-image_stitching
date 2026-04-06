## 📖 Overview
This project implements an image stitching pipeline from scratch to generate seamless panoramic images.</br>
The approach is based on classical computer vision techniques including feature detection, feature matching, and homography estimation.</br>
</br>
The goal is to align multiple overlapping images and blend them into a single coherent panorama.

## 💡 Key Concepts
### 🔹 Feature Detection & Matching
 - Detect keypoints using feature detectors (e.g., SIFT, ORB)
 - Match keypoints between overlapping images
 - Filter matches to retain reliable correspondences
### 🔹 Homography Estimation

We estimate a projective transformation that aligns two images.

### $x' = Hx$
 - H: 3×3 homography matrix
 - Maps points from one image to another
 - Estimated using matched feature correspondences (e.g., RANSAC)
### 🔹 Perspective Warping
Apply homography to warp one image into the coordinate frame of another</br>
Align overlapping regions</br>
### 🔹 Image Blending
Combine aligned images into a seamless panorama</br>
Reduce visible seams using blending techniques</br>
## ⚙️ Pipeline
<ol>
  <li>Input two or more overlapping images</li>
  <li>Detect keypoints and extract descriptors</li>
  <li>Match features across images</li>
  <li>Estimate homography using robust methods (e.g., RANSAC)</li>
  <li>Warp images using the estimated transformation</li>
  <li>Blend images to produce the final panorama</li></li>
</ol>


## 🧪 Results
 - Successfully generated panoramic images from overlapping inputs
 - Achieved alignment through homography-based transformation
 - Blending reduces visible seams between images

## 📊 Conclusion
This project demonstrates a complete classical image stitching pipeline using geometric transformations and feature-based alignment.</br>
It highlights how local feature correspondences can be leveraged to estimate global image transformations and generate panoramas.
