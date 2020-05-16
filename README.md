# Optical-Flow
A multi-feature tracker that takes in object bounding boxes for the first frame and tracks them over the remaining frames

**Team Members:** Arnav Dhamija and Saumya Shah

## Running the code

```
python3 getFirstFrame.py vids/Easy.mp4
```
This gets the first frame png 


```
labelImg first.png
```
Create the bounding box for the first frame adn save the points

```
./create_video.sh
```
Creates an AVI video of the object being tracking using optical flow of the features

## Results

1. Tracking of 2 cars driving straight

![result_easy](/results/easy_compressed.gif)

2. Tracking of a car in low lighting conditions with significant turning of the vehicle

![result_easy](/results/medium.gif)

## Features

1. The features are detected from the first frame and tracked using optical flow estimation
2. The features are refreshed regularly when the count drops, this helps in cases when the object changes significantly overtime as in case of second example.
3. A transformation is applied depending on the movement of the features to calculate the bounding box position, scale and orientation.
4. Iterative refinement is performed for precise feature flow estimation.

