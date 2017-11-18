#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

  VectorXd rmse(estimations[0].size());
  rmse.fill(0);
  for (int i = 0; i < estimations.size(); i++) {
    auto diff = estimations[i] - ground_truth[i];
    VectorXd diffSq = diff.array() * diff.array();
    rmse += diffSq;
  }

  return rmse / (double) estimations.size();

}