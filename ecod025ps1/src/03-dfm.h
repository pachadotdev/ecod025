/* Dynamic Factor Model (DFM) with Kalman Smoothing */

// Standardize data: mean 0 and common variance
std::tuple<Mat<double>, rowvec, double> dfm_standardize_(const Mat<double>& X) {
  int T = X.n_rows;
  int N = X.n_cols;
  
  rowvec means = mean(X, 0);
  Mat<double> X_centered = X.each_row() - means;
  
  double total_var = accu(square(X_centered)) / (T * N);
  double common_sd = sqrt(total_var);
  
  Mat<double> X_std = X_centered / common_sd;
  
  return std::make_tuple(X_std, means, common_sd);
}

// Unstandardize data back to original scale
Mat<double> dfm_unstandardize_(const Mat<double>& X_std, const rowvec& means, double common_sd) {
  Mat<double> X_unstd = X_std * common_sd;
  X_unstd.each_row() += means;
  return X_unstd;
}

// PCA initialization for factors and loadings
std::pair<Mat<double>, Mat<double>> dfm_pca_init_(const Mat<double>& X, int n_factors) {
  Mat<double> U, V;
  vec s;
  svd_econ(U, s, V, X.t());
  
  Mat<double> Lambda = U.cols(0, n_factors-1);
  Mat<double> F = X * Lambda;
  
  // Normalize factors to have unit variance
  vec F_var = var(F, 0, 0).t();
  F.each_col() /= sqrt(F_var);
  Lambda.each_col() %= sqrt(F_var);
  
  return std::make_pair(F, Lambda);
}

// Kalman Filter and Smoother for DFM
struct KalmanOutput {
  Mat<double> F_smooth;     // Smoothed factors (T x state_dim)
  Cube<double> P_smooth;    // Smoothed covariances (state_dim x state_dim x T)
  Cube<double> PPm_smooth;  // Lag-one smoothed covariances for EM
  double loglik;
};

KalmanOutput dfm_kalman_smoother_(const Mat<double>& X, const Mat<double>& Lambda,
                                  const Mat<double>& A, const Mat<double>& Q,
                                  const Mat<double>& R, int n_factors, int p) {
  int T = X.n_rows;
  int N = X.n_cols;
  int state_dim = (p > 0) ? n_factors * p : n_factors;
  
  // Storage for filter outputs
  Mat<double> F_pred(T, state_dim);
  Mat<double> F_filt(T, state_dim);
  Cube<double> P_pred(state_dim, state_dim, T);
  Cube<double> P_filt(state_dim, state_dim, T);
  
  // Observation matrix H
  Mat<double> H = zeros<Mat<double>>(N, state_dim);
  H.submat(0, 0, N-1, n_factors-1) = Lambda;
  
  // Initial conditions
  vec F0 = zeros<vec>(state_dim);
  Mat<double> P0 = eye<Mat<double>>(state_dim, state_dim) * 10.0;
  
  double loglik = 0.0;
  double log_2pi = log(2 * datum::pi);
  
  // Forward pass (Kalman filter)
  for (int t = 0; t < T; ++t) {
    // Prediction
    if (t == 0) {
      F_pred.row(t) = F0.t();
      P_pred.slice(t) = P0;
    } else {
      F_pred.row(t) = (A * F_filt.row(t-1).t()).t();
      P_pred.slice(t) = A * P_filt.slice(t-1) * A.t() + Q;
      // Ensure symmetry
      P_pred.slice(t) = 0.5 * (P_pred.slice(t) + P_pred.slice(t).t());
    }
    
    // Update
    vec y_t = X.row(t).t();
    vec y_pred = H * F_pred.row(t).t();
    vec innov = y_t - y_pred;
    
    Mat<double> S = H * P_pred.slice(t) * H.t() + R;
    S = 0.5 * (S + S.t()); // Ensure symmetry
    
    Mat<double> K = P_pred.slice(t) * H.t() * inv(S);
    
    F_filt.row(t) = F_pred.row(t) + (K * innov).t();
    P_filt.slice(t) = (eye<Mat<double>>(state_dim, state_dim) - K * H) * P_pred.slice(t);
    P_filt.slice(t) = 0.5 * (P_filt.slice(t) + P_filt.slice(t).t());
    
    // Log-likelihood
    double detS = det(S);
    if (detS > 0) {
      double mahal = dot(innov, solve(S, innov));
      loglik -= 0.5 * (N * log_2pi + log(detS) + mahal);
    }
  }
  
  // Backward pass (Smoother)
  Mat<double> F_smooth(T, state_dim);
  Cube<double> P_smooth(state_dim, state_dim, T);
  Cube<double> PPm_smooth(state_dim, state_dim, T);
  
  // Initialize with filtered values at T
  F_smooth.row(T-1) = F_filt.row(T-1);
  P_smooth.slice(T-1) = P_filt.slice(T-1);
  
  // Backward recursion
  for (int t = T-2; t >= 0; --t) {
    Mat<double> J = P_filt.slice(t) * A.t() * inv(P_pred.slice(t+1));
    
    F_smooth.row(t) = F_filt.row(t) + 
                      (J * (F_smooth.row(t+1) - F_pred.row(t+1)).t()).t();
    
    P_smooth.slice(t) = P_filt.slice(t) + 
                        J * (P_smooth.slice(t+1) - P_pred.slice(t+1)) * J.t();
    P_smooth.slice(t) = 0.5 * (P_smooth.slice(t) + P_smooth.slice(t).t());
    
    // Lag-one covariance for EM (Cov(F_t+1, F_t | Y))
    if (t == T-2) {
      Mat<double> K_T = P_filt.slice(T-1) * H.t() * 
                        inv(H * P_pred.slice(T-1) * H.t() + R);
      PPm_smooth.slice(t+1) = (eye<Mat<double>>(state_dim, state_dim) - K_T * H) * 
                              A * P_filt.slice(t);
    } else {
      Mat<double> J_next = P_filt.slice(t+1) * A.t() * inv(P_pred.slice(t+2));
      PPm_smooth.slice(t+1) = P_filt.slice(t+1) * J.t() + 
                              J_next * (PPm_smooth.slice(t+2) - A * P_filt.slice(t+1)) * J.t();
    }
  }
  
  // First lag-one covariance
  if (T > 1) {
    Mat<double> J0 = P0 * A.t() * inv(P_pred.slice(0));
    PPm_smooth.slice(0) = P_filt.slice(0) * J0.t() + 
                          (P_filt.slice(0) * A.t() * inv(P_pred.slice(1))) * 
                          (PPm_smooth.slice(1) - A * P_filt.slice(0)) * J0.t();
  }
  
  KalmanOutput out;
  out.F_smooth = F_smooth.cols(0, n_factors-1);  // Extract actual factors
  out.P_smooth = P_smooth;
  out.PPm_smooth = PPm_smooth;
  out.loglik = loglik;
  
  return out;
}

// EM algorithm for DFM
struct DFMParams {
  Mat<double> Lambda;
  Mat<double> A;
  Mat<double> Q;
  Mat<double> R;
};

DFMParams dfm_em_algorithm_(const Mat<double>& X, int n_factors, int p, 
                            int max_iter = 100, double tol = 1e-6) {
  int T = X.n_rows;
  int N = X.n_cols;
  int state_dim = (p > 0) ? n_factors * p : n_factors;
  
  // Initialize with PCA
  auto pca_init = dfm_pca_init_(X, n_factors);
  Mat<double> F = pca_init.first;
  Mat<double> Lambda = pca_init.second;
  
  // Initialize dynamics
  Mat<double> A, Q;
  if (p > 0) {
    Mat<double> A_var = var_estimate_(as_doubles_matrix(F), p, false);
    A = var_companion_(A_var, p, n_factors, false);
    
    auto fitted_resid = var_fitted_resid_(F, A_var, p, false);
    Mat<double> eta = fitted_resid.second;
    
    Q = zeros<Mat<double>>(state_dim, state_dim);
    Q.submat(0, 0, n_factors-1, n_factors-1) = cov(eta);
  } else {
    A = eye<Mat<double>>(n_factors, n_factors) * 0.9;
    Q = eye<Mat<double>>(n_factors, n_factors) * 0.1;
  }
  
  // Initialize R
  Mat<double> X_fitted = F * Lambda.t();
  Mat<double> residuals = X - X_fitted;
  Mat<double> R = diagmat(var(residuals, 0, 0).t());
  
  double prev_loglik = -datum::inf;
  
  // EM iterations
  for (int iter = 0; iter < max_iter; ++iter) {
    // E-step: Run Kalman smoother
    KalmanOutput ks = dfm_kalman_smoother_(X, Lambda, A, Q, R, n_factors, p);
    
    // Check convergence
    if (iter > 0 && std::abs(ks.loglik - prev_loglik) / std::abs(prev_loglik) < tol) {
      break;
    }
    prev_loglik = ks.loglik;
    
    // M-step: Update parameters
    
    // Sufficient statistics
    Mat<double> delta = zeros<Mat<double>>(N, n_factors);
    Mat<double> gamma = zeros<Mat<double>>(n_factors, n_factors);
    Mat<double> beta = zeros<Mat<double>>(n_factors, n_factors);
    Mat<double> gamma1 = zeros<Mat<double>>(n_factors, n_factors);
    
    for (int t = 0; t < T; ++t) {
      rowvec f_t = ks.F_smooth.row(t);
      Mat<double> P_t = ks.P_smooth.slice(t).submat(0, 0, n_factors-1, n_factors-1);
      
      delta += X.row(t).t() * f_t;
      gamma += f_t.t() * f_t + P_t;
      
      if (t > 0) {
        rowvec f_tm1 = ks.F_smooth.row(t-1);
        Mat<double> PPm_t = ks.PPm_smooth.slice(t).submat(0, 0, n_factors-1, n_factors-1);
        beta += f_t.t() * f_tm1 + PPm_t;
        gamma1 += f_tm1.t() * f_tm1 + 
                  ks.P_smooth.slice(t-1).submat(0, 0, n_factors-1, n_factors-1);
      }
    }
    
    // Update Lambda
    Lambda = delta.t() * inv(gamma);
    Lambda = Lambda.t();
    
    // Update R
    Mat<double> X_pred = ks.F_smooth * Lambda.t();
    Mat<double> resid = X - X_pred;
    vec R_diag = sum(square(resid), 0).t() / T;
    
    // Add trace term for measurement error variance
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < n_factors; ++j) {
        double trace_term = Lambda(i, j) * Lambda(i, j) * 
                           (gamma(j, j) / T - 
                            as_scalar(ks.F_smooth.col(j).t() * ks.F_smooth.col(j)) / T);
        R_diag(i) += trace_term;
      }
    }
    R = diagmat(R_diag);
    
    // Update A and Q (only for dynamic model)
    if (p > 0) {
      // Update A (only top-left block for companion form)
      Mat<double> A_new = beta * inv(gamma1);
      A.submat(0, 0, n_factors-1, n_factors-1) = A_new;
      
      // Update Q (only top-left block)
      Mat<double> Q_new = (gamma - beta * A_new.t() - A_new * beta.t() + 
                          A_new * gamma1 * A_new.t()) / T;
      Q_new = 0.5 * (Q_new + Q_new.t());
      Q.submat(0, 0, n_factors-1, n_factors-1) = Q_new;
    }
  }
  
  DFMParams params;
  params.Lambda = Lambda;
  params.A = A;
  params.Q = Q;
  params.R = R;
  
  return params;
}

// DFM forecasting
Mat<double> dfm_forecast_(const Mat<double>& F_last, const Mat<double>& Lambda,
                         const Mat<double>& A, int n_factors, int p, int h) {
  int N = Lambda.n_rows;
  int state_dim = (p > 0) ? n_factors * p : n_factors;
  
  Mat<double> F_forecast(h, n_factors);
  Mat<double> X_forecast(h, N);
  
  // Initialize state vector
  vec state = zeros<vec>(state_dim);
  if (p > 0) {
    // Fill companion form state
    int max_lags = std::min(p, static_cast<int>(F_last.n_rows));
    for (int i = 0; i < max_lags; ++i) {
      state.subvec(i * n_factors, (i + 1) * n_factors - 1) = 
        F_last.row(F_last.n_rows - 1 - i).t();
    }
  } else {
    state = F_last.row(F_last.n_rows - 1).t();
  }
  
  // Generate forecasts
  for (int i = 0; i < h; ++i) {
    state = A * state;
    F_forecast.row(i) = state.subvec(0, n_factors - 1).t();
    X_forecast.row(i) = (Lambda * F_forecast.row(i).t()).t();
  }
  
  return X_forecast;
}

/* roxygen
@title Dynamic Factor Model (DFM)
@description Estimates a DFM using EM algorithm with Kalman smoothing
@param x Matrix of observed variables (T x N)
@param n_factors Number of latent factors
@param p VAR lag order for factor dynamics
@param max_iter Maximum EM iterations
@param tol Convergence tolerance
@param forecast_h Forecast horizon
@export
*/
[[cpp4r::register]] list dfm_model(const doubles_matrix<>& x, int n_factors, 
                                   int p = 1, int max_iter = 100, 
                                   double tol = 1e-6, int forecast_h = 0) {
  Mat<double> X_raw = as_Mat(x);
  int T = X_raw.n_rows;
  int N = X_raw.n_cols;
  
  if (T <= p) {
    throw std::runtime_error("Sample size must be greater than lag order p");
  }
  
  if (n_factors >= N) {
    throw std::runtime_error("Number of factors must be less than number of variables");
  }

  // Standardize data
  auto standardize_result = dfm_standardize_(X_raw);
  Mat<double> X = std::get<0>(standardize_result);
  rowvec means = std::get<1>(standardize_result);
  double common_sd = std::get<2>(standardize_result);
  
  // Estimate DFM parameters
  DFMParams params = dfm_em_algorithm_(X, n_factors, p, max_iter, tol);
  
  // Final smoothing pass
  KalmanOutput final_ks = dfm_kalman_smoother_(X, params.Lambda, params.A, 
                                               params.Q, params.R, n_factors, p);
  
  // Compute fitted values
  Mat<double> X_fitted_std = final_ks.F_smooth * params.Lambda.t();
  Mat<double> X_fitted = dfm_unstandardize_(X_fitted_std, means, common_sd);
  Mat<double> residuals = X_raw - X_fitted;
  
  // Information criteria
  int n_params = N * n_factors + n_factors * n_factors * p + 
                 n_factors * (n_factors + 1) / 2 + N;
  double aic = -2 * final_ks.loglik + 2 * n_params;
  double bic = -2 * final_ks.loglik + n_params * log(T);
  
  // Prepare results
  writable::list result;
  
  result.push_back({"factors"_nm = as_doubles_matrix(final_ks.F_smooth)});
  result.push_back({"loadings"_nm = as_doubles_matrix(params.Lambda)});
  result.push_back({"transition"_nm = as_doubles_matrix(params.A)});
  result.push_back({"factor_cov"_nm = as_doubles_matrix(params.Q)});
  result.push_back({"idiosync_cov"_nm = as_doubles_matrix(params.R)});
  result.push_back({"fitted_values"_nm = as_doubles_matrix(X_fitted)});
  result.push_back({"residuals"_nm = as_doubles_matrix(residuals)});
  result.push_back({"loglik"_nm = cpp4r::as_sexp(final_ks.loglik)});
  result.push_back({"aic"_nm = cpp4r::as_sexp(aic)});
  result.push_back({"bic"_nm = cpp4r::as_sexp(bic)});
  result.push_back({"n_factors"_nm = cpp4r::as_sexp(n_factors)});
  result.push_back({"lag_order"_nm = cpp4r::as_sexp(p)});
  result.push_back({"n_variables"_nm = cpp4r::as_sexp(N)});
  result.push_back({"n_obs"_nm = cpp4r::as_sexp(T)});
  
  // Metrics
  result.push_back({"rmsfe"_nm = cpp4r::as_sexp(rmsfe(X_raw.col(0), X_fitted.col(0)))});
  result.push_back({"mae"_nm = cpp4r::as_sexp(mae(X_raw.col(0), X_fitted.col(0)))});
  
  // Forecasts
  if (forecast_h > 0) {
    Mat<double> forecasts_std = dfm_forecast_(final_ks.F_smooth, params.Lambda, 
                                              params.A, n_factors, p, forecast_h);
    Mat<double> forecasts = dfm_unstandardize_(forecasts_std, means, common_sd);
    result.push_back({"forecasts"_nm = as_doubles_matrix(forecasts)});
    result.push_back({"forecast_horizon"_nm = cpp4r::as_sexp(forecast_h)});
  }
  
  return result;
}
