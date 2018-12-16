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
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // this code is from the lecture exercise:
  if ((estimations.size() == 0) || (estimations.size() != ground_truth.size()))  {
    return rmse;
  }

  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i) {
    VectorXd delta  =  estimations[i] - ground_truth[i];
    delta = delta.array() * delta.array();
    rmse += delta;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    /**
     TODO:
     * Calculate a Jacobian here.
     */
  // this part of the code is from lectures quizes
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  //let's check for division by zero
  if((fabs(c1) < 0.0001) || (fabs(c2) < 0.0001) || (fabs(c3) < 0.0001)){
    return Hj;
  }

  //compute Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
         -(py/c1), (px/c1), 0, 0,
         py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
