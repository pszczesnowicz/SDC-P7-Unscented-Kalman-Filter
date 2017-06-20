#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE (const vector<VectorXd> &estimations,
                               const vector<VectorXd> &ground_truth) {
  
  VectorXd root_mean_squared_error(4);
  root_mean_squared_error << 0, 0, 0, 0;
  
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    cout << "CalculateRMSE() - Error: Estimation and ground truth vectors not equal in size.";
    return root_mean_squared_error;
  }
  
  for(int i = 0; i < estimations.size(); ++i) {
    VectorXd squared_residuals = pow((estimations[i] - ground_truth[i]).array(), 2.0);
    root_mean_squared_error += squared_residuals;
  }
  
  // Calculating the mean.
  root_mean_squared_error = root_mean_squared_error / estimations.size();
  
  // Calculating the squared root.
  root_mean_squared_error = root_mean_squared_error.cwiseSqrt();
  
  return root_mean_squared_error;
  
}

double Tools::NormalizeAngle (double angle) {
  angle = fmod(angle + M_PI, 2.0 * M_PI);
  
  if (angle < 0) {
    angle += 2.0 * M_PI;
  }
  
  return angle - M_PI;
}


double Tools::CalculateNIS (VectorXd z_diff, MatrixXd S) {
  return (z_diff.transpose() * S.inverse() * z_diff)(0, 0);
}
