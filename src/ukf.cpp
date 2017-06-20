#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Class constructor
UKF::UKF() {
  
  is_initialized_ = false;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  // State dimension.
  n_x_ = 5;
  
  // Augmented state dimension.
  n_aug_ = 7;
  
  // Sigma point spreading parameter.
  lambda_ = 3.0 - n_aug_;

  // State vector.
  x_ = VectorXd(n_x_);

  // Covariance matrix.
  P_ = MatrixXd(n_x_, n_x_);

  // Weights vector.
  weights_ = VectorXd(2 * n_aug_ + 1);
  
  // Predicted sigma points matrix.
  X_sig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
}

// Class destructor
UKF::~UKF() {}


// @param {MeasurementPackage} meas_package The latest measurement data of either radar or laser.
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  // ******************
  // * Initialization *
  // ******************
  
  if (!is_initialized_) {
    
    // Initializing the state vector.
    x_.fill(0.0);

    // Initializes the state vector x & y positions using the first LASER measurements of x and y.
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    }
    
    // Initializing the state vector x & y positions using the first RADAR measurements of rho and phi.
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      x_(0) = meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1));
      x_(1) = meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1));
    }
    
    // Initializing the covariance matrix.
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    
    // Initializing the weights vector.
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
      weights_(i) = 0.5 / (lambda_ + n_aug_);
    }
    
    time_us_ = meas_package.timestamp_;
    
    is_initialized_ = true;
    
    return;
  }

  // *******************
  // * Prediction step *
  // *******************
  
  // Computing the elapsed time in seconds.
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  
  time_us_ = meas_package.timestamp_;
  
  // Predicting the state vector and covariance matrix of non-simultaneous measurements.
  if (delta_t > 0) {
      Prediction(delta_t);
  }

  // ***************
  // * Update step *
  // ***************
  
  // Updating state and covariance using LASER measurements.
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    if (!use_laser_) {
      return;
    }
    UpdateLidar(meas_package);
  }
  
  // Updating state and covariance using RADAR measurements.
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    if (!use_radar_) {
      return;
    }
    UpdateRadar(meas_package);
  }
}


// Predicts sigma points, the state, and the state covariance matrix.
// @param {double} delta_t the change in time (in seconds) between the last measurement and this one.
void UKF::Prediction(double delta_t) {
  
  Tools tools;
  
  // *************************************
  // * Generating augmented sigma points *
  // *************************************
  
  // Creating the augmented mean state.
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  
  // Creating the process noise vector.
  MatrixXd Q = MatrixXd(2, 2);
  Q <<  pow(std_a_, 2.0), 0,
        0, pow(std_yawdd_, 2.0);
  
  // Creating the augmented covariance matrix.
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;
  
  // Creating the augmented square root of the covariance matrix.
  MatrixXd A_aug = P_aug.llt().matrixL();
  
  // Creating the augmented sigma point matrix.
  MatrixXd X_sig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  X_sig_aug.fill(0.0);
  X_sig_aug.col(0) = x_aug;
  
  // Pre-computing the augmented sigma point spread.
  MatrixXd spread_aug = sqrt(lambda_ + n_aug_) * A_aug;
  
  for (int i = 0; i < n_aug_; ++i) {
    X_sig_aug.col(i + 1) = x_aug + spread_aug.col(i);
    X_sig_aug.col(i + 1 + n_aug_) = x_aug - spread_aug.col(i);
  }
  
  // ***************************
  // * Predicting sigma points *
  // ***************************
  
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    
    VectorXd state_change = VectorXd(n_x_);
    state_change.fill(0.0);
    
    if (fabs(X_sig_aug(4, i)) < 0.001)  {
      state_change(0) = X_sig_aug(2, i) * cos(X_sig_aug(3, i)) * delta_t;
      state_change(1) = X_sig_aug(2, i) * sin(X_sig_aug(3, i)) * delta_t;
    }
    else {
      state_change(0) = (X_sig_aug(2, i) / X_sig_aug(4, i)) * (sin(X_sig_aug(3, i) + X_sig_aug(4, i) * delta_t) - sin(X_sig_aug(3, i)));
      state_change(1) = (X_sig_aug(2, i) / X_sig_aug(4, i)) * (-cos(X_sig_aug(3, i) + X_sig_aug(4, i) * delta_t) + cos(X_sig_aug(3, i)));
      state_change(3) = X_sig_aug(4, i) * delta_t;
    }
    
    VectorXd process_noise = VectorXd(n_x_);
    
    process_noise(0) = 0.5 * pow(delta_t, 2.0) * cos(X_sig_aug(3, i)) * X_sig_aug(5, i);
    process_noise(1) = 0.5 * pow(delta_t, 2.0) * sin(X_sig_aug(3, i)) * X_sig_aug(5, i);
    process_noise(2) = delta_t * X_sig_aug(5, i);
    process_noise(3) = 0.5 * pow(delta_t, 2.0) * X_sig_aug(6, i);
    process_noise(4) = delta_t * X_sig_aug(6, i);
    
    X_sig_pred_.col(i) = X_sig_aug.col(i).head(5) + state_change + process_noise;
    
  }
  
  // ************************************************
  // * Generating the predicted mean and covariance *
  // ************************************************
  
  x_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ += weights_(i) * X_sig_pred_.col(i);
  }
  
  P_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    
    VectorXd x_diff = X_sig_pred_.col(i) - x_;
    x_diff(3) = tools.NormalizeAngle(x_diff(3));

    P_ += weights_(i) * x_diff * x_diff.transpose();
    
  }
  
}


// Updates the state and the state covariance matrix using a laser measurement.
// @param {MeasurementPackage} meas_package
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  
  Tools tools;
  
  // **************************************************
  // * Predicting the measurement mean and covariance *
  // **************************************************
  
  // Measurement dimension
  int n_z = 2;
  
  // Creating the predicted, measurement space, sigma points matrix.
  MatrixXd Z_sig_pred = MatrixXd(n_z, 2 * n_aug_ + 1);
  Z_sig_pred.fill(0.0);
  
  // Creating the predicted measurement mean vector.
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    
    Z_sig_pred.col(i) = X_sig_pred_.col(i).head(n_z);
    
    z_pred += weights_(i) * Z_sig_pred.col(i);
    
  }
  
  // Creating the predicted measurement covariance matrix.
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Z_sig_pred.col(i) - z_pred;
   
    S += weights_(i) * z_diff * z_diff.transpose();
    
  }
  
  // Creating the measurement noise matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_laspx_, 0,
        0, std_laspy_;
  
  S += R;
  
  // ******************************************
  // * Updating the state mean and covariance *
  // ******************************************
  
  // Creating the cross correlation matrix.
  MatrixXd T = MatrixXd(n_x_, n_z);
  T.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    T += weights_(i) * (X_sig_pred_.col(i) - x_) * (Z_sig_pred.col(i) - z_pred).transpose();
  }
  
  // Creating the Kalman gain matrix.
  MatrixXd K = T * S.inverse();
  
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  
  x_ += K * z_diff;
  
  P_ -= K * S * K.transpose();
  
  // ***********************
  // * Calculating the NIS *
  // ***********************
  
  NIS_las_ = tools.CalculateNIS(z_diff, S);
  
  // cout << NIS_las_ << "\n";

}


// Updates the state and the state covariance matrix using a radar measurement.
// @param {MeasurementPackage} meas_package
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  
  Tools tools;
  
  // **************************************************
  // * Predicting the measurement mean and covariance *
  // **************************************************

  // Measurement dimension
  int n_z = 3;
  
  // Creating the predicted measurement space sigma points matrix.
  MatrixXd Z_sig_pred = MatrixXd(n_z, 2 * n_aug_ + 1);
  Z_sig_pred.fill(0.0);
  
  // Creating the predicted measurement mean vector.
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    
    // Rho
    Z_sig_pred(0, i) = sqrt(pow(X_sig_pred_(0, i), 2.0) + pow(X_sig_pred_(1,i), 2.0));
    
    // Phi
    if (fabs(X_sig_pred_(0, i)) < 0.001 && fabs(X_sig_pred_(1, i) < 0.001)) {
      Z_sig_pred(1, i) = 0;
    }
    else {
      Z_sig_pred(1, i) = atan2(X_sig_pred_(1, i), X_sig_pred_(0, i));
    }
    
    // Rho dot
    if (fabs(Z_sig_pred(0, i)) < 0.001) {
      Z_sig_pred(2, i) = 0;
    }
    else {
      Z_sig_pred(2, i) = (X_sig_pred_(0, i) * cos(X_sig_pred_(3, i)) * X_sig_pred_(2, i) + X_sig_pred_(1, i) * sin(X_sig_pred_(3, i)) * X_sig_pred_(2, i)) / Z_sig_pred(0, i);
    }
    
    z_pred += weights_(i) * Z_sig_pred.col(i);
    
  }
  
  // Creating the predicted measurement covariance matrix.
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Z_sig_pred.col(i) - z_pred;
    z_diff(1) = tools.NormalizeAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
    
  }
  
  // Creating the measurement noise matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_radr_, 0, 0,
        0, std_radphi_, 0,
        0, 0, std_radrd_;
  
  S += R;
  
  // ******************************************
  // * Updating the state mean and covariance *
  // ******************************************
  
  // Creating the cross correlation matrix.
  MatrixXd T = MatrixXd(n_x_, n_z);
  T.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    VectorXd z_diff = Z_sig_pred.col(i) - z_pred;
    z_diff(1) = tools.NormalizeAngle(z_diff(1));
    
    VectorXd x_diff = X_sig_pred_.col(i) - x_;
    x_diff(1) = tools.NormalizeAngle(x_diff(1));
    
    T += weights_(i) * x_diff * z_diff.transpose();
    
  }
  
  // Creating the Kalman gain matrix.
  MatrixXd K = T * S.inverse();
  
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  z_diff(1) = tools.NormalizeAngle(z_diff(1));
  
  x_ += K * z_diff;
  
  P_ -= K * S * K.transpose();
  
  // ***********************
  // * Calculating the NIS *
  // ***********************
  
  NIS_rad_ = tools.CalculateNIS(z_diff, S);
  
  // cout << NIS_rad_ << "\n";
  
}
