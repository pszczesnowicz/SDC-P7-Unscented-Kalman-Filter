#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:

  // Constructor.
  Tools();

  // Destructor.
  virtual ~Tools();

  // A helper method to calculate RMSE.
  VectorXd CalculateRMSE (const vector<VectorXd> &estimations,
                         const vector<VectorXd> &ground_truth);
  
  // Normalizes angle to -pi and pi.
  double NormalizeAngle (double angle);
  
  // Calculates the Normalized Innovation Squared.
  double CalculateNIS (VectorXd z_diff, MatrixXd S);

};

#endif /* TOOLS_H_ */
