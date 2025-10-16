/* Dynamic Factor Model (DFM) */

// Standardize data: mean 0 and common variance (to avoid scaling issues)
// Returns: standardized data, column means, and common std dev
std::tuple<Mat<double>, rowvec, double> dfm_standardize_(const Mat<double>& X) {
  int T = X.n_rows;
  int N = X.n_cols;
  
  // Center each variable and save means
  rowvec means = mean(X, 0);
  Mat<double> X_centered = X.each_row() - means;
  
  // Compute common variance (pooled across all variables)
  double total_var = accu(square(X_centered)) / (T * N);
  double common_sd = sqrt(total_var);
  
  // Standardize to common variance
  Mat<double> X_std = X_centered / common_sd;
  
  return std::make_tuple(X_std, means, common_sd);
}

// Unstandardize data back to original scale
Mat<double> dfm_unstandardize_(const Mat<double>& X_std, const rowvec& means, double common_sd) {
  // X_unstd = X_std * common_sd + means (broadcast means across rows)
  Mat<double> X_unstd = X_std * common_sd;
  X_unstd.each_row() += means;
  return X_unstd;
}

// PCA-based factor extraction for initialization
std::pair<Mat<double>, Mat<double>> dfm_extract_factors_(const Mat<double>& X, int n_factors) {
  // X should already be standardized
  
  // Perform SVD on X' (N x T matrix)
  Mat<double> U, V;
  vec s;
  svd_econ(U, s, V, X.t());
  
  // Factor loadings (first n_factors principal components)
  Mat<double> Lambda = U.cols(0, n_factors-1);
  
  // Common factors (T x n_factors)
  Mat<double> F = X * Lambda;
  
  return std::make_pair(F, Lambda);
}

// Initialize DFM parameters using PCA
std::tuple<Mat<double>, Mat<double>, Mat<double>, Mat<double>> dfm_init_(
    const Mat<double>& X, int n_factors, int p) {
  int T = X.n_rows;
  
  // Step 1: Extract factors using PCA
  auto factor_result = dfm_extract_factors_(X, n_factors);
  Mat<double> F = factor_result.first;
  Mat<double> Lambda = factor_result.second;
  
  // Step 2: Estimate factor dynamics F_t = A * F_{t-1} + eta_t
  // Convert VAR(p) to VAR(1) in companion form
  Mat<double> A;
  Mat<double> Q;
  
  if (p > 0) {
    // Estimate VAR(p) for factors
    Mat<double> A_var = var_estimate_(as_doubles_matrix(F), p, false);
    
    // Create companion form matrix
    A = var_companion_(A_var, p, n_factors, false);
    
    // Compute residual covariance
    auto fitted_resid = var_fitted_resid_(F, A_var, p, false);
    Mat<double> eta = fitted_resid.second;
    
    // Q is for the companion form (only top-left block is non-zero)
    int state_dim = n_factors * p;
    Q = zeros<Mat<double>>(state_dim, state_dim);
    Q.submat(0, 0, n_factors-1, n_factors-1) = (eta.t() * eta) / eta.n_rows;
  } else {
    // Static factor model: F_t = constant (no dynamics)
    A = eye<Mat<double>>(n_factors, n_factors);
    Q = eye<Mat<double>>(n_factors, n_factors) * 0.01;
  }
  
  // Initialize R as diagonal (idiosyncratic errors)
  Mat<double> X_fitted = F.cols(0, n_factors-1) * Lambda.t();
  Mat<double> residuals = X - X_fitted;
  Mat<double> R = diagmat(sum(square(residuals), 0) / T);
  
  return std::make_tuple(Lambda, A, Q, R);
}

// Kalman Filter for DFM state-space model
std::tuple<Mat<double>, Mat<double>, double> dfm_kalman_(
    const Mat<double>& X, const Mat<double>& Lambda, const Mat<double>& A,
    const Mat<double>& Q, const Mat<double>& R, int n_factors, int p) {
  
  int T = X.n_rows;
  int N = X.n_cols;
  int state_dim = (p > 0) ? n_factors * p : n_factors;
  
  // State variables
  Mat<double> F_pred(T, state_dim);  // Predicted states
  Mat<double> F_upd(T, state_dim);   // Updated states
  Mat<double> P_pred(state_dim, state_dim);  // Predicted covariance
  Mat<double> P_upd(state_dim, state_dim);   // Updated covariance
  
  // Log likelihood
  double loglik = 0.0;
  
  // Observation matrix H: maps state to observables
  // X_t = H * F_t + u_t
  Mat<double> H = zeros<Mat<double>>(N, state_dim);
  H.submat(0, 0, N-1, n_factors-1) = Lambda;  // Only first n_factors matter
  
  // Transition matrix (A is already in companion form if p > 1)
  Mat<double> F_trans = A;
  
  // Process noise covariance (Q is already in companion form)
  Mat<double> Q_full = Q;
  
  // Initial conditions: diffuse prior
  vec mu0 = zeros<vec>(state_dim);
  Mat<double> P0 = eye<Mat<double>>(state_dim, state_dim) * 10.0;  // Diffuse
  
  // Kalman filter loop
  for (int t = 0; t < T; ++t) {
    // Prediction step
    if (t == 0) {
      F_pred.row(t) = mu0.t();
      P_pred = P0;
    } else {
      F_pred.row(t) = (F_trans * F_upd.row(t-1).t()).t();
      P_pred = F_trans * P_upd * F_trans.t() + Q_full;
    }
    
    // Update step
    vec y_t = X.row(t).t();
    vec y_pred = H * F_pred.row(t).t();
    vec innov = y_t - y_pred;
    
    Mat<double> S = H * P_pred * H.t() + R;
    Mat<double> K = P_pred * H.t() * inv(S);
    
    F_upd.row(t) = F_pred.row(t) + (K * innov).t();
    P_upd = (eye<Mat<double>>(state_dim, state_dim) - K * H) * P_pred;
    
    // Log likelihood contribution
    double log_det_S = log(det(S));
    double mahal = dot(innov, solve(S, innov));
    loglik += -0.5 * (N * log(2 * datum::pi) + log_det_S + mahal);
  }
  
  // Extract just the factors (first n_factors columns of state)
  Mat<double> F_smooth = F_upd.cols(0, n_factors-1);
  
  return std::make_tuple(F_smooth, F_upd, loglik);
}

// EM Algorithm for DFM parameter estimation
std::tuple<Mat<double>, Mat<double>, Mat<double>, Mat<double>> dfm_em_(
    const Mat<double>& X, int n_factors, int p, int max_iter = 50, double tol = 1e-4) {
  int T = X.n_rows;
  
  // Initialize parameters
  auto init_params = dfm_init_(X, n_factors, p);
  Mat<double> Lambda = std::get<0>(init_params);
  Mat<double> A = std::get<1>(init_params);
  Mat<double> Q = std::get<2>(init_params);
  Mat<double> R = std::get<3>(init_params);
  
  double prev_loglik = -datum::inf;
  
  // EM iterations
  for (int iter = 0; iter < max_iter; ++iter) {
    // E-step: Kalman filter to get expected factors
    auto kalman_result = dfm_kalman_(X, Lambda, A, Q, R, n_factors, p);
    Mat<double> F_smooth = std::get<0>(kalman_result);
    double loglik = std::get<2>(kalman_result);
    
    // M-step: Update parameters given expected factors
    
    // Update Lambda: X_t = Lambda * F_t + u_t
    // Lambda_new = (sum_t X_t F_t') * (sum_t F_t F_t')^{-1}
    Lambda = (X.t() * F_smooth) * inv(F_smooth.t() * F_smooth);
    
    // Update R (diagonal idiosyncratic covariance)
    Mat<double> residuals = X - F_smooth * Lambda.t();
    R = diagmat(sum(square(residuals), 0) / T);
    
    // Update A and Q (factor dynamics)
    if (p > 0) {
      // Re-estimate VAR(p) for updated factors
      Mat<double> A_var = var_estimate_(as_doubles_matrix(F_smooth), p, false);
      A = var_companion_(A_var, p, n_factors, false);
      
      // Update Q
      auto factor_resid = var_fitted_resid_(F_smooth, A_var, p, false);
      Mat<double> eta = factor_resid.second;
      
      int state_dim = n_factors * p;
      Q = zeros<Mat<double>>(state_dim, state_dim);
      Q.submat(0, 0, n_factors-1, n_factors-1) = (eta.t() * eta) / eta.n_rows;
    }
    // For static model (p=0), A and Q remain fixed
    
    // Check convergence
    if (iter > 0 && abs(loglik - prev_loglik) < tol) {
      break;
    }
    prev_loglik = loglik;
  }
  
  return std::make_tuple(Lambda, A, Q, R);
}

// DFM forecasting using state-space representation
Mat<double> dfm_forecast_(const Mat<double>& F_state, const Mat<double>& Lambda,
                         const Mat<double>& A, int n_factors, int p, int h) {
  int N = Lambda.n_rows;
  
  // Initialize forecasts
  Mat<double> F_forecast(h, n_factors);
  Mat<double> X_forecast(h, N);
  
  // Current state (last row of F_state which is in companion form)
  Mat<double> F_current = F_state.row(F_state.n_rows - 1);
  
  // Forecast loop
  for (int i = 0; i < h; ++i) {
    // Predict next state: F_{t+h} = A^h * F_t
    F_current = (A * F_current.t()).t();
    
    // Extract actual factors (first n_factors elements of state)
    F_forecast.row(i) = F_current.cols(0, n_factors - 1);
    
    // Forecast observables: X_{t+h} = Lambda * F_{t+h}
    X_forecast.row(i) = (Lambda * F_forecast.row(i).t()).t();
  }
  
  return X_forecast;
}

/* roxygen
@title Dynamic Factor Model (DFM)
@description Estimates a Dynamic Factor Model using PCA initialization and EM algorithm with Kalman filter.
@param x Matrix of observed variables (T x N)
@param n_factors Number of latent factors (r < N)
@param p VAR lag order for factor dynamics (0 for static model, recommend p=1 for small samples)
@param max_iter Maximum EM iterations
@param tol Convergence tolerance for log-likelihood
@param forecast_h Forecast horizon (0 for no forecast)
@export
*/
[[cpp4r::register]] list dfm_model(const doubles_matrix<>& x, int n_factors, int p = 1,
                                   int max_iter = 50, double tol = 1e-4, int forecast_h = 0) {
  Mat<double> X_raw = as_Mat(x);
  int T = X_raw.n_rows;
  int N = X_raw.n_cols;
  
  if (T <= p) {
    throw std::runtime_error("Sample size must be greater than lag order p");
  }
  
  if (n_factors >= N) {
    throw std::runtime_error("Number of factors must be less than number of variables");
  }

  // Standardize data to mean 0 and common variance
  auto standardize_result = dfm_standardize_(X_raw);
  Mat<double> X = std::get<0>(standardize_result);
  rowvec means = std::get<1>(standardize_result);
  double common_sd = std::get<2>(standardize_result);
  
  // Estimate DFM parameters using EM algorithm
  auto em_result = dfm_em_(X, n_factors, p, max_iter, tol);
  Mat<double> Lambda = std::get<0>(em_result);
  Mat<double> A = std::get<1>(em_result);
  Mat<double> Q = std::get<2>(em_result);
  Mat<double> R = std::get<3>(em_result);
  
  // Final Kalman filter to get smoothed factors
  auto kalman_result = dfm_kalman_(X, Lambda, A, Q, R, n_factors, p);
  Mat<double> F = std::get<0>(kalman_result);
  Mat<double> F_state = std::get<1>(kalman_result);
  double loglik = std::get<2>(kalman_result);
  
  // Compute fitted values on standardized scale
  Mat<double> X_fitted_std = F * Lambda.t();
  Mat<double> residuals_std = X - X_fitted_std;
  
  // Unstandardize fitted values to original scale
  Mat<double> X_fitted = dfm_unstandardize_(X_fitted_std, means, common_sd);
  Mat<double> residuals = X_raw - X_fitted;
  
  // Information criteria
  int n_params = N * n_factors +  // Lambda
                 n_factors * n_factors * p +  // A (in companion form)
                 n_factors * n_factors +  // Q
                 N;  // R (diagonal)
  double aic = aic_metric(residuals, n_params);
  
  // Prepare results list
  writable::list result;
  
  result.push_back({"factors"_nm = as_doubles_matrix(F)});
  result.push_back({"loadings"_nm = as_doubles_matrix(Lambda)});
  result.push_back({"transition"_nm = as_doubles_matrix(A)});
  result.push_back({"factor_cov"_nm = as_doubles_matrix(Q)});
  result.push_back({"idiosync_cov"_nm = as_doubles_matrix(R)});

  result.push_back({"fitted_values"_nm = as_doubles_matrix(X_fitted)});
  result.push_back({"residuals"_nm = as_doubles_matrix(residuals)});
  result.push_back({"loglik"_nm = cpp4r::as_sexp(loglik)});
  result.push_back({"n_factors"_nm = cpp4r::as_sexp(n_factors)});
  result.push_back({"lag_order"_nm = cpp4r::as_sexp(p)});

  result.push_back({"n_variables"_nm = cpp4r::as_sexp(N)});
  result.push_back({"n_obs"_nm = cpp4r::as_sexp(T)});
  
  // Compute RMSFE and MAE for first variable (ORIGINAL scale)
  result.push_back({"rmsfe"_nm = cpp4r::as_sexp(rmsfe(X_raw.col(0), X_fitted.col(0)))});
  result.push_back({"mae"_nm = cpp4r::as_sexp(mae(X_raw.col(0), X_fitted.col(0)))});
  result.push_back({"aic"_nm = cpp4r::as_sexp(aic)});
  
  // Add forecasts if requested
  if (forecast_h > 0) {
    // For forecasting, construct initial companion state from last p observations of F
    // F is T x n_factors, we need to create companion state [F_T, F_{T-1}, ..., F_{T-p+1}]
    Mat<double> F_init_state;
    if (p > 0) {
      int state_dim = n_factors * p;
      F_init_state = zeros<Mat<double>>(1, state_dim);
      for (int i = 0; i < p; ++i) {
        int t_idx = T - 1 - i;  // T-1, T-2, ..., T-p
        if (t_idx >= 0) {
          F_init_state.cols(i * n_factors, (i + 1) * n_factors - 1) = F.row(t_idx);
        }
      }
    } else {
      F_init_state = F.row(T - 1);  // Just the last factor for static model
    }
    
    Mat<double> forecasts_std = dfm_forecast_(F_init_state, Lambda, A, n_factors, p, forecast_h);
    
    // Unstandardize forecasts to original scale
    Mat<double> forecasts = dfm_unstandardize_(forecasts_std, means, common_sd);
    
    result.push_back({"forecasts"_nm = as_doubles_matrix(forecasts)});
    result.push_back({"forecast_horizon"_nm = cpp4r::as_sexp(forecast_h)});
  }
  
  return result;
}
