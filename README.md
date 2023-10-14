# FaceDetectionAndTracking
Face and Hand detection and Tracking
In this project, I have detected and tracked face and hands using track-by-detection approach. Initially I detected face in the video uploaded using Viola-Jones Algorithm and this is intial point to detect and track hands and shoulders.

//Face detection and tracking using Viola-Jones algorithm 

The code uses the Viola-Jones algorithm, implemented through a Haar-like features-based cascade classifier provided by OpenCV, to detect faces in a grayscale image. It iterates through the list of detected faces and returns the coordinates and dimensions of the last detected face and has drawn rectangles around the detected faces.

//Hand detection and tracking using MediaPipe 

To track and detect hands I have used MediaPipe methods like process(). Initially converted the colorspace to RGB as expected by the process method of MediaPipe and optionally drawn bounding boxes to visualize the detected hands. 

//Shoulder detection and tracking 

the code detects and tracks shoulders based on contrast information within a region defined around the face.I have used gradient analysis, linear regression, and weighted averaging to estimate the shoulder positions and then tracked.
