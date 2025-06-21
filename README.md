# RadarProcessing Project
In this project I aimed to test a hybrid CNN and MUSIC/MVDR (classical Direction of Arrival (DOA) estimation algorithms) approach to remain full angular resolution and accuracy but improving the computation time.
The hybrid approach includes a CNN pre-prediction of a range of angles where the DOA lays in and a precise sweep over these angles using MUSIC or MVDR algorithms to find the exact DOA
The results showed the CNN to be capable to detect the rough DOA from the Covariance Matrix R, with some difficulties in angular resolution 
The time saved for single targets is estimated to be around 80%
