#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;
  
  // State dimension
  int n_x_;
  
  // Augmented state dimension
  int n_aug_;
  
  // Laser measurement dimension.
  int n_las_z_;
  
  // Radar measurement dimension.
  int n_rad_z_;
  
  // Sigma point spreading parameter
  double lambda_;
  
  // State vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  // State covariance matrix
  MatrixXd P_;

  // Predicted sigma points matrix
  MatrixXd X_sig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;
  
  // Laser measurement noise matrix
  MatrixXd R_las_;
  
  // Radar measurement noise matrix
  MatrixXd R_rad_;
  
  // Laser measurement function.
  MatrixXd H_las_;

  // Weights of sigma points
  VectorXd weights_;
  
  // Laser Normalized Innovation Squared (NIS)
  double NIS_las_;
  
  // Radar Normalized Innovation Squared (NIS)
  double NIS_rad_;
  
  // Constructor
  UKF();

  // Destructor
  virtual ~UKF();

  // ProcessMeasurement
  // @param meas_package The latest measurement data of either radar or laser
  void ProcessMeasurement(MeasurementPackage meas_package);

  // Prediction Predicts sigma points, the state, and the state covariance matrix
  // @param delta_t Time between k and k+1 in s
  void Prediction(double delta_t);

  // Updates the state and the state covariance matrix using a laser measurement
  // Uses the linear Kalman Filter equations
  // @param meas_package The measurement at k+1
  void UpdateLidarKF(MeasurementPackage meas_package);
  
  // Updates the state and the state covariance matrix using a laser measurement
  // Uses the Unscented Kalman Filter equations
  // @param meas_package The measurement at k+1
  void UpdateLidarUKF(MeasurementPackage meas_package);

  // Updates the state and the state covariance matrix using a radar measurement
  // @param meas_package The measurement at k+1
  void UpdateRadarUKF(MeasurementPackage meas_package);
  
};

#endif /* UKF_H */
