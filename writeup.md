# Writeup: Track 3D-Objects Over Time

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?

Step 1, Extended Kalman Filter:
In this step I implemented an extended kalman filter. This involved defining the predict and update processes, and deriving the parameters required to support these steps, such as a model of motion (F) and prediction noise (Q).
This filter allowed the system to track a single target, which it did with a RMSE value of 0.32.

Step 2, Track Management:
In this step I implemented track management functions to perform track initialization, scoring, deletion, and status change. This involved handling coordinate transformations and defining rules for how the score and status of tracks should be determined.
This allowed the system to initialize a track it hadn't seen before, and delete it when it was no longer receiving measurement updates. A y offset in the lidar sensor input data used meant the RMSE achieved was underhwelming at 0.78.

Step 3, Association:
In this step I implemented an association process, which paired incoming measurements with existing tracks using a simple nearest neighbour approach. The Mahalanobis distance was used to ensure the pairing decision accounted for positional uncertainty, and gating was used to eliminate obviously incorrect potential pairs. Unnasigned tracks and measurements were kept track of.
This allowed the system to track multiple targets at once, and it did so with an average RMSE of ~0.15

Step 4, Camera Fusion:
In this step I incorperated camera data into the system to perform sensor fusion. This required transforming 3d state positions into 2d camera image coordinates, initializing the camera sensor data object, and writing a fov checking function.
This allowed for camera measurements to fused with lidar measurements via the kalman filter process, providing an improved RMSE of ~0.13.

I found the association step the hardest to complete. I initially didn't realise that the unassigned_track/meas lists were supposed to just contain the indices of tracks and measurements, and that determination of these indices was supposed to occur in get_closest_track_and_meas() and not associate().

### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 
Absolutely. Firstly, the simple act of adding more measurements reduces the uncertainty and increases the accuracy of state information. In addition, by diversifying sensors you add redundancy (in case of sensor failure), and allow weaknesses to be compensated for (e.g. cameras can see low reflectivity objects, where lidar performs poorly). There is also potential for the total field of view can be increased.

Comparing the RMSE plots from step 3 and 4, you can see that the addition of the camera sensor decreased the average RMSE of objects previously tracked by lidar alone. You can also see that the camera sensor spotted track 11 consitently enough to confirm it (if briefly), which wasn't accomplished when using lidar alone.

### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
Hardware is a major challenge. I observed in this project that the program ran slower when camera measurements were incorperated. Given that objects must be tracked in near real-time, this means the minimum processing specs required are higher for fusion systems than camera or lidar only. Of course there is also the additional required hardware of the extra sensors themselves, plus the addition work of calibration.

### 4. Can you think of ways to improve your tracking results in the future?
In the association step, more advanced techniques such as probabilistic data association could be employed to improve performance when faced with challenging and ambiguous decisions.
The model of motion could be tweaked to improve prediction performance, by incorperating information like acceleration, the constrained degrees of freedom of motion for vehicles, or even environmental factors (like the path of the lane).
The track scoring parameters could be fine tuned via experimentation to provide the best results for circumstances of interest.

