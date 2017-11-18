#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 10;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 10;

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

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;
  use_laser_ = true;
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (double) (lambda_ + n_aug_);
  for (int i = 1; i < weights_.size(); i++) {
      weights_(i) = 1 / (2.0 * (lambda_ + n_aug_));
  }

  std_a_ = 1;
  std_yawdd_ = 1;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    // If initializing, set state to measurements
    if (meas_package.sensor_type_ == meas_package.RADAR) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_d = meas_package.raw_measurements_(2);
      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
      x_(2) = 1;
      x_(3) = rho;
      x_(4) = 1;
    }
    else {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      x_(2) = 1;
      x_(3) = 1;
      x_(4) = 1;
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    return;
  }

  // Update
  // 1. Predict
  // 2. Adjust
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.;
  time_us_ = meas_package.timestamp_;
  Prediction(dt);
  if (meas_package.sensor_type_ == meas_package.RADAR) {
    UpdateRadar(meas_package);
  }
  else {
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // Choose the sigma points
  MatrixXd X_sig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
  MatrixXd offset = P_aug.llt().matrixL();
  offset *= sqrt(lambda_ + n_aug_);
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  X_sig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    X_sig_aug.col(i + 1) = x_aug + offset.col(i);
    X_sig_aug.col(n_aug_ + i + 1) = x_aug - offset.col(i);
  }

  // Predict sigma points
  Xsig_pred_ = MatrixXd(n_x_, X_sig_aug.cols());
  for (int j = 0; j < Xsig_pred_.cols(); j++) {
    VectorXd x_aug = X_sig_aug.col(j);
    VectorXd xk = x_aug.head(n_x_);
    
    double px = x_aug(0);
    double py = x_aug(1);
    double vel = x_aug(2);
    double yaw = x_aug(3);
    double yaw_d = x_aug(4);
    double nu_a = x_aug(n_x_);
    double nu_yawdd = x_aug(n_x_ + 1);
    
    // Noise vector
    VectorXd process_noise(n_x_);
    process_noise << 
    0.5 * delta_t * delta_t * cos(yaw) * nu_a,
    0.5 * delta_t * delta_t * sin(yaw) * nu_a,
    delta_t * nu_a,
    0.5 * delta_t * delta_t * nu_yawdd,
    delta_t * nu_yawdd;
    
    // Noise-free prediction
    VectorXd pred(n_x_);
    if (yaw_d == 0) {
        pred(0) = vel * cos(yaw) * delta_t;
        pred(1) = vel * sin(yaw) * delta_t;
    }
    else {
        pred(0) = vel / yaw_d * (sin(yaw + yaw_d * delta_t) - sin(yaw));
        pred(1) = vel / yaw_d * (-cos(yaw + yaw_d * delta_t) + cos(yaw));
    }
    pred(2) = 0;
    pred(3) = yaw_d * delta_t;
    pred(4) = 0;
    
    Xsig_pred_.col(j) = xk + pred + process_noise;
  }

  // From predicted sigma points, predict mean state (x_)   and compute the state covariance matrix P_
  x_ = Xsig_pred_ * weights_;
  MatrixXd centered = Xsig_pred_;
  MatrixXd weighed_centered = centered;
  for (int j = 0; j < centered.cols(); j++) {
      centered.col(j) -= x_;
      weighed_centered.col(j) = centered.col(j) * weights_(j);
  }
  P_ = weighed_centered * centered.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // Transform sigma points to measurement space
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, Xsig_pred_.cols());
  for (int j = 0;  j < Xsig_pred_.cols(); j++) {
    VectorXd col = Xsig_pred_.col(j);
    double px = col(0);
    double py = col(1);
    Zsig.col(j) << px, py;
  }
  VectorXd z_pred = Zsig * weights_;

  // Covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int j = 0; j < Zsig.cols(); j++)
  {
    VectorXd zm = Zsig.col(j);
    S += (Zsig.col(j) - z_pred) * (Zsig.col(j) - z_pred).transpose() * weights_(j);
  }
  // Add measurement noise covariance matrix to bottom right of covariance matrix S
  MatrixXd R(2, 2);
  R << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;
  S += R;

  // UKF update
  MatrixXd Tc = MatrixXd(n_x_, meas_package.raw_measurements_.size());
  Tc.fill(0);
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd z_diff = (Zsig.col(i) - z_pred);
    Tc +=  (Xsig_pred_.col(i) - x_) * z_diff.transpose() * weights_(i);
  }
  MatrixXd K = Tc * S.inverse();
  x_ += K * (meas_package.raw_measurements_ - z_pred);
  P_ -= K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // Transform sigma points to radar space
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, Xsig_pred_.cols());
  for (int j = 0; j < Xsig_pred_.cols(); j++) {
    VectorXd col = Xsig_pred_.col(j);
    double px = col(0);
    double py = col(1);
    double vel = col(2);
    double yaw = col(3);
    double yaw_d = col(4);
    double rho = sqrt(px * px + py * py);
    double phi = atan2(py, px);
    double rho_d = (px * cos(yaw) * vel + py * sin(yaw) * vel) / rho;
    VectorXd z(3);
    z << rho, phi, rho_d;
    Zsig.col(j) = z;
  }
  //Mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  //Covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int j = 0; j < Zsig.cols(); j++)
  {
    VectorXd zm = Zsig.col(j);
    while (zm(2) > M_PI)
      zm(2) -= 2. * M_PI;
    while (zm(2) < -M_PI)
      zm(2) += 2. * M_PI;
    S += (Zsig.col(j) - z_pred) * (Zsig.col(j) - z_pred).transpose() * weights_(j);
  }

  // Add measurement noise covariance to covariance matrix S
  MatrixXd R(3, 3);
  R << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;
  S += R;

  // UKF update
  MatrixXd Tc = MatrixXd(n_x_, meas_package.raw_measurements_.size());
  Tc.fill(0);
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
    // Don't for get to normalize the angle in the differences...
    VectorXd z_diff = (Zsig.col(i) - z_pred);
    while (z_diff(1) > M_PI) z_diff(1) -= 2 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2 * M_PI; 
    Tc +=  (Xsig_pred_.col(i) - x_) * z_diff.transpose() * weights_(i);
  }
  MatrixXd K = Tc * S.inverse();
  x_ += K * (meas_package.raw_measurements_ - z_pred);
  P_ -= K * S * K.transpose();

}
