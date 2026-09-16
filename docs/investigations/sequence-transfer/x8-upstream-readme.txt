This README file was generated on [2024-11-20] (2024-11-20) by [Bogdan Løw-Hansen].
Last updated: [2024-11-20].

-------------------
GENERAL INFORMATION
-------------------
// Title of Dataset: System identification campaign - Skywalker X8 UAV
// DOI: 10.18710/U4TL
// Contact Information
<The person to be contacted for questions about the dataset>
     // Name: Bogdan Løw-Hansen
     // Institution: NTNU
     // Email: bogdan.l.hansen@ntnu.no
     // ORCID: [0000-0003-0462-2811](https://orcid.org/0000-0003-0462-2811)

// Contributors: NTNU UAV Lab.
// Data Type: Flight logs.
// Date of Collection: 2023-05-08.
// Geographic location: Agdenes Airfield, Norway.
// Funding sources: Research Council of Norway: IKTPLUSS project 316425.

// Description of dataset: 
<The data was collected by the NTNU UAV lab as part of the system identification campaign for the Skywalker X8 unmanned aerial vehicle (UAV) in May of 2023 at Agdenes Airfield, Norway.>

--------------------------
METHODOLOGICAL INFORMATION
--------------------------
// Description of sources and methods used for collection/generation of data: 
<The dataset has three sources:
1) External inertial measurement unit (IMU) STIM300. The measurements have been transformed to the body frame with origin in center of gravity (CG).
     1) linear accelerations
     2) angular velocities
2) Ardupilot autopilot running on Cube Orange hardware.  
	1) Estimated states from the Ardupilot extended Kalman filter (EKF). The autopilot hardware was placed close to the CG of the UAV. The states estimated by the autopilot EKF are therefore assumed to be about the CG.
	2) GPS data
	3) Propulsion system data consisting of battery voltage and current
3) Single board computer (SBC) used to generate system identification input commands for the UAV control surfaces (left and right elevon).>

<The ardupilot EKF doesn't provide an estimate for the vertical wind speed. 
The vertical wind speed was instead estimated based on the difference between the measured airspeed and the estimated relative airspeed in the horizontal plane. 
The resulting angle of attack was then validated based on a previous flight with a five-hole probe presented in [1].
The indicated speed in the dataset is the corrected airspeed computed from the estimated inertial velocity and the estimated wind velocity, which includes the estimated vertical wind.>

<[1] K. T. Borup, T. I. Fossen, T. A. Johansen, [A Machine Learning Approach for Estimating Air Data Parameters](https://folk.ntnu.no/torarnj/AES_ML.pdf), IEEE Transactions on Aerospace and Electronic Systems, vol. 56, pp. 2157-2173, 2020; https://doi.org/10.1109/TAES.2019.2945383>


// Methods for processing the data: 
<The final maneuvers were obtained by synchronizing and resampling the data to 40Hz from the three sources mentioned above. 
The original IMU data was sampled at 200Hz; however, it was downsampled to 40Hz to match the EKF data sampling rate. 
GPS and the propulsion system data were upsampled from 10Hz to 40Hz. The SBC data was originally sampled at 40Hz.
The synchronization was performed manually. The STIM300 IMU data was synchronized with the Cube orange autopilot IMU data. 
The SBC elevon inputs were synchronized based on autopilot mode switch information, i.e. (auto, stabilize, manual) which is recorded by the SBC and the Ardupilot log. 
The manual synchronization of the SBC input commands can have an effect on the computed time delay for the actuator models.>

// Environmental/experimental conditions: 
<The data was collected by the NTNU UAV lab as part of the system identification campaign for the Skywalker X8 UAV in May of 2023 at Agdenes Airfield, Norway. 
There was a strong north-west wind during the experiments with a significant vertical component.>
  
--------------------
DATA & FILE OVERVIEW
--------------------
// File List: 
< 
- The dataset includes 17 maneuvers, where each maneuver is approximately 10 seconds long. 
- The maneuvers are split into a training (13) and a validation (4) set.
- The file names indicate the type of the recorded maneuver.
- The original files are provided in the ".mat" format; a supplemental set in the ".csv" format is provided as well in the "interoperability_files" folder.>

// Is this an updated version of a dataset published on DataverseNO? no
// Version number of dataset:     
// File name: 
// Why was the file updated? 
// When was the file updated (YYYY-MM-DD)?: 
// What was changed? 

-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: [*.mat]
-----------------------------------------
The data streams in the maneuver arrays are concatenated as column vectors. 
Each maneuver has one time array and 40 data arrays. The data names are self-explanatory. 

| Index | Variable                  | Unit       | Description                                   |
| ----- | ------------------------- | ---------  | ------------------------------                |
| 1     | t                         | s          | time                                          |
| 2     | elevator                  | rad        |                                               |
| 3     | aileron                   | rad        |                                               |
| 4     | throttle                  | -          | Normalized throttle command in range [0,1]    |
| 5     | IMU_acc_x                 | m/s^2      | Body frame acceleration in CG                 |
| 6     | IMU_acc_y                 | m/s^2      | Body frame acceleration in CG                 |
| 7     | IMU_acc_z                 | m/s^2      | Body frame acceleration in CG                 |
| 8     | IMU_angvel_p              | rad/s      | Body frame gyro measurement                   |
| 9     | IMU_angvel_q              | rad/s      | Body frame gyro measurement                   |
| 10    | IMU_angvel_r              | rad/s      | Body frame gyro measurement                   |
| 11    | EstimatedState_phi        | rad        |                                               |
| 12    | EstimatedState_theta      | rad        |                                               |
| 13    | EstimatedState_psi        | rad        |                                               |
| 14    | EstimatedState_p          | rad/s      |                                               |
| 15    | EstimatedState_q          | rad/s      |                                               |
| 16    | EstimatedState_r          | rad/s      |                                               |
| 17    | EstimatedState_u          | m/s        |                                               |
| 18    | EstimatedState_v          | m/s        |                                               |
| 19    | EstimatedState_w          | m/s        |                                               |
| 20    | EstimatedState_vx         | m/s        | Inertial velocity in NED                      |
| 21    | EstimatedState_vy         | m/s        | Inertial velocity in NED                      |
| 22    | EstimatedState_vz         | m/s        | Inertial velocity in NED                      |
| 23    | EstimatedStreamVelocity_x | m/s        | Estimated wind velocity in NED                |
| 24    | EstimatedStreamVelocity_y | m/s        | Estimated wind velocity in NED                |
| 25    | EstimatedStreamVelocity_z | m/s        | Estimated wind velocity in NED                |
| 26    | EstimatedState_alpha      | rad        |                                               |
| 27    | EstimatedState_beta       | rad        |                                               |
| 28    | TrueSpeed                 | m/s        |                                               |
| 29    | IndicatedSpeed            | m/s        |                                               |
| 30    | GPS_lon                   | degree     |                                               |
| 31    | GPS_lat                   | degree     |                                               |
| 32    | GPS_height                | m          |                                               |
| 33    | GPS_x                     | m          |                                               |
| 34    | GPS_y                     | m          |                                               |
| 35    | GPS_z                     | m          |                                               |
| 36    | GPS_cog                   | rad        |                                               |
| 37    | GPS_sog                   | m/s        |                                               |
| 38    | Current                   | A          |                                               |
| 39    | Voltage                   | V          |                                               |
| 40    | Temperature               | degree C   |                                               |
| 41    | Pressure                  | mbar       |                                               |


--------------------------
SHARING/ACCESS INFORMATION
--------------------------
// Licenses/Restrictions: See Terms tab.
// Links to publications that cite or use the data: See metadata field Related Publication.
// Recommended citation: See citation generated by repository.
