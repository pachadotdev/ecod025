/*
==============================================
Dynamic Factor Model (DFM)
==============================================
*/

// Simple PCA factor extraction (similar to FA-VAR)
std::pair<Mat<double>, Mat<double>> dfm_extract_factors_(const Mat<double>& X, int n_factors) {
  // Center the data
  Mat<double> X_centered = X.each_row() - mean(X, 0);
  
  // Perform SVD on X' (N x T matrix)
  Mat<double> U, V;
  vec s;
  svd_econ(U, s, V, X_centered.t());
  
  // Factor loadings (first n_factors principal components)
  Mat<double> Lambda = U.cols(0, n_factors-1);
  
  // Common factors (T x n_factors)
  Mat<double> F = X_centered * Lambda;
  
  return std::make_pair(F, Lambda);
}

// Initialize DFM parameters using PCA and simple VAR
std::tuple<Mat<double>, Mat<double>, Mat<double>, Mat<double>> dfm_init_(
    const Mat<double>& X, int n_factors, int p) {
  
  // Step 1: Extract factors using PCA
  auto factor_result = dfm_extract_factors_(X, n_factors);
  Mat<double> F = factor_result.first;
  Mat<double> Lambda = factor_result.second;
  
  // Step 2: Estimate factor VAR (reuse existing functions)
  Mat<double> A;
  Mat<double> Q;
  
  if (p > 0) {
    // Use same approach as FA-VAR for factor dynamics
    A = var_estimate_(as_doubles_matrix(F), p, true);
    
    // Compute residual covariance
    auto fitted_resid = var_fitted_resid_(F, A, p, true);
    Mat<double> eta = fitted_resid.second;
    Q = (eta.t() * eta) / (eta.n_rows - A.n_rows);
  } else {
    // Static factor model: no dynamics, just return mean
    A = mean(F, 0).t();  // Column vector of factor means
    Q = eye<Mat<double>>(n_factors, n_factors) * 0.1;
  }
  
  // Initialize R as diagonal
  Mat<double> X_fitted = F * Lambda.t();
  Mat<double> residuals = X - X_fitted;
  Mat<double> R = diagmat(sum(square(residuals), 0) / X.n_rows);
  
  return std::make_tuple(Lambda, A, Q, R);
}

// Kalman Filter for DFM
std::tuple<Mat<double>, Mat<double>, Mat<double>> dfm_kalman_(
    const Mat<double>& X, const Mat<double>& Lambda, const Mat<double>& A,
    const Mat<double>& Q, const Mat<double>& R, int n_factors, int p) {
  
  int T = X.n_rows;
  int N = X.n_cols;
  
  // Initialize state variables
  Mat<double> F_pred(T, n_factors * std::max(p, 1));  // Predicted factors
  Mat<double> F_upd(T, n_factors * std::max(p, 1));   // Updated factors
  Mat<double> P_pred(n_factors * std::max(p, 1), n_factors * std::max(p, 1)); // Predicted covariance
  Mat<double> P_upd(n_factors * std::max(p, 1), n_factors * std::max(p, 1));  // Updated covariance
  
  // Log likelihood
  double loglik = 0.0;
  
  // Observation matrix H (maps states to factors)
  Mat<double> H = zeros<Mat<double>>(N, n_factors * std::max(p, 1));
  H.submat(0, 0, N-1, n_factors-1) = Lambda;
  
  // Transition matrix F_trans
  Mat<double> F_trans;
  if (p > 0) {
    F_trans = var_companion_(A, p, n_factors, true);
  } else {
    F_trans = eye<Mat<double>>(n_factors, n_factors);
  }
  
  // Process noise covariance
  Mat<double> Q_full = zeros<Mat<double>>(n_factors * std::max(p, 1), n_factors * std::max(p, 1));
  Q_full.submat(0, 0, n_factors-1, n_factors-1) = Q;
  
  // Initial conditions
  vec mu0 = zeros<vec>(n_factors * std::max(p, 1));
  Mat<double> P0 = eye<Mat<double>>(n_factors * std::max(p, 1), n_factors * std::max(p, 1));
  
  // Kalman filter loop
  for (int t = 0; t < T; ++t) {
    if (t == 0) {
      F_pred.row(t) = mu0.t();
      P_pred = P0;
    } else {
      // Prediction step
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
    P_upd = (eye<Mat<double>>(P_pred.n_rows, P_pred.n_cols) - K * H) * P_pred;
    
    // Log likelihood
    loglik += -0.5 * (N * log(2 * datum::pi) + log(det(S)) + dot(innov, solve(S, innov)));
  }
  
  // Extract just the factors (first n_factors columns)
  Mat<double> F_smooth = F_upd.cols(0, n_factors-1);
  
  return std::make_tuple(F_smooth, F_upd, loglik * arma::ones<Mat<double>>(1, 1));
}

// Simplified EM Algorithm for DFM parameter estimation
std::tuple<Mat<double>, Mat<double>, Mat<double>, Mat<double>> dfm_em_(
    const Mat<double>& X, int n_factors, int p, int max_iter = 50, double tol = 1e-4) {
  
  int T = X.n_rows;
  
  // Initialize parameters (simplified)
  auto init_params = dfm_init_(X, n_factors, p);
  Mat<double> Lambda = std::get<0>(init_params);
  Mat<double> A = std::get<1>(init_params);
  Mat<double> Q = std::get<2>(init_params);
  Mat<double> R = std::get<3>(init_params);
  
  double prev_loglik = -datum::inf;
  
  // Simplified EM iterations
  for (int iter = 0; iter < max_iter; ++iter) {
    // E-step: Kalman filter and smoother
    auto kalman_result = dfm_kalman_(X, Lambda, A, Q, R, n_factors, p);
    Mat<double> F_smooth = std::get<0>(kalman_result);
    double loglik = std::get<2>(kalman_result)(0, 0);
    
    // M-step: Simplified parameter updates
    
    // Update Lambda (factor loadings) - simplified
    Lambda = solve(F_smooth.t() * F_smooth, F_smooth.t() * X).t();
    
    // Update R (diagonal only) - simplified  
    Mat<double> residuals = X - F_smooth * Lambda.t();
    R = diagmat(sum(square(residuals), 0) / T);
    
    // Update A and Q (factor dynamics) - reuse VAR functions
    if (p > 0) {
      A = var_estimate_(as_doubles_matrix(F_smooth), p, true);
      auto factor_resid = var_fitted_resid_(F_smooth, A, p, true);
      Mat<double> eta = factor_resid.second;
      Q = (eta.t() * eta) / (eta.n_rows - A.n_rows);
    } else {
      // Static model: update mean
      A = mean(F_smooth, 0).t();
      // Keep Q constant for static model
    }
    
    // Simple convergence check
    if (iter > 0 && abs(loglik - prev_loglik) < tol) {
      break;
    }
    prev_loglik = loglik;
  }
  
  return std::make_tuple(Lambda, A, Q, R);
}

// DFM forecasting using state-space representation
Mat<double> dfm_forecast_(const Mat<double>& F, const Mat<double>& Lambda,
                         const Mat<double>& A, const Mat<double>& Q, 
                         int n_factors, int p, int h) {
  
  int T = F.n_rows;
  int N = Lambda.n_rows;

  int A_r = A.n_rows;
  int A_c = A.n_cols;
  
  // Initialize forecasts
  Mat<double> F_forecast(h, n_factors);
  Mat<double> X_forecast(h, N);
  
  // Get initial conditions for factors
  Mat<double> F_init = F.rows(T - std::max(p, 1), T - 1);
  
  // Create transition matrix for factor dynamics
  Mat<double> F_trans;
  if (p > 0) {
    F_trans = var_companion_(A, p, n_factors, true);
  } else {
    F_trans = eye<Mat<double>>(n_factors, n_factors);
  }
  
  // Forecast factors using VAR dynamics
  Mat<double> F_state;
  if (p > 1) {
    // Initialize with companion form state
    F_state = vectorise(F_init.t()).t();
  } else if (p == 1) {
    // For p=1, just use last observation
    F_state = F.row(T - 1);
  }
  
  for (int i = 0; i < h; ++i) {
    if (p > 0) {
      // Predict next period factors using transition matrix
      Mat<double> F_pred_state = (F_trans * F_state.t()).t();
      
      // Extract actual factors (first n_factors columns)
      Mat<double> F_pred = F_pred_state.cols(0, n_factors - 1);
      F_forecast.row(i) = F_pred;
      
      // Update state for next iteration
      F_state = F_pred_state;
    } else {
      // Static factor model - use constant (mean from A)
      if (A_r == n_factors && A_c == 1) {
        F_forecast.row(i) = A.t();  // Use mean as constant
      } else {
        F_forecast.row(i) = F.row(T - 1);  // Fallback to last observation
      }
    }
    
    // Forecast observables using factor loadings
    X_forecast.row(i) = (Lambda * F_forecast.row(i).t()).t();
  }
  
  return X_forecast;
}

/* roxygen
@title Main DFM estimation function
@export
*/
[[cpp4r::register]] list dfm_model(const doubles_matrix<>& x, int n_factors, int p = 1,
                                   bool include_const = true, int max_iter = 25, double tol = 1e-4,
                                   int forecast_h = 0, int n_lags = 0) {
  Mat<double> X_raw = as_Mat(x);
  int T = X_raw.n_rows;
  int N_raw = X_raw.n_cols;
  
  Mat<double> X;
  int N;
  
  // If univariate (N=1) and n_lags specified, create lagged matrix
  if (N_raw == 1 && n_lags > 0) {
    Mat<double> X_lagged = create_lags_(X_raw, n_lags);
    // Join original with lags: [X_t, X_{t-1}, ..., X_{t-n_lags}]
    Mat<double> X_contemp = X_raw.rows(n_lags, T - 1);
    X = join_horiz(X_contemp, X_lagged);
    T = X.n_rows;
    N = X.n_cols;
  } else {
    X = X_raw;
    N = N_raw;
  }
  
  if (T <= p) {
    throw std::runtime_error("Sample size must be greater than lag order p");
  }
  
  if (n_factors >= N) {
    throw std::runtime_error("Number of factors must be less than number of variables");
  }
  
  // Estimate DFM parameters using EM algorithm
  auto em_result = dfm_em_(X, n_factors, p, max_iter, tol);
  Mat<double> Lambda = std::get<0>(em_result);
  Mat<double> A = std::get<1>(em_result);
  Mat<double> Q = std::get<2>(em_result);
  Mat<double> R = std::get<3>(em_result);
  
  // Final Kalman filter to get factors
  auto kalman_result = dfm_kalman_(X, Lambda, A, Q, R, n_factors, p);
  Mat<double> F = std::get<0>(kalman_result);
  double loglik = std::get<2>(kalman_result)(0, 0);
  
  // Compute fitted values and residuals
  Mat<double> X_fitted = F * Lambda.t();
  Mat<double> residuals = X - X_fitted;
  
  // Information criteria
  int n_params = N * n_factors + n_factors * n_factors * p + n_factors + N;  // Lambda + A + Q + R
  double aic = aic_metric(residuals, n_params);  // SSR-based AIC
  
  // Prepare results list
  writable::list result(12);

  result[0] = as_doubles_matrix(F);
  result[1] = as_doubles_matrix(Lambda);
  result[2] = as_doubles_matrix(A);
  result[3] = as_doubles_matrix(Q);
  result[4] = as_doubles_matrix(R);
  result[5] = as_doubles_matrix(X_fitted);
  result[6] = as_doubles_matrix(residuals);
  result[7] = cpp4r::as_sexp(loglik);
  result[8] = cpp4r::as_sexp(n_factors);
  result[9] = cpp4r::as_sexp(p);
  result[10] = cpp4r::as_sexp(N);
  result[11] = cpp4r::as_sexp(T);

  result.names() = {"factors", "loadings", "transition", "factor_cov", "idiosync_cov",
                    "fitted_values", "residuals", "loglik",
                    "n_factors", "lag_order", "n_variables", "n_obs"};
  
  // Add forecasts if requested
  if (forecast_h > 0) {
    Mat<double> forecasts = dfm_forecast_(F, Lambda, A, Q, n_factors, p, forecast_h);
    result.push_back({"forecasts"_nm = as_doubles_matrix(forecasts)});
    result.push_back({"forecast_horizon"_nm = cpp4r::as_sexp(forecast_h)});
  }

  // Add model metrics (only for first variable/column which is the contemporaneous value)
  result.push_back({"rmsfe"_nm = cpp4r::as_sexp(rmsfe(X.col(0), X_fitted.col(0)))});
  result.push_back({"mae"_nm = cpp4r::as_sexp(mae(X.col(0), X_fitted.col(0)))});
  result.push_back({"aic"_nm = cpp4r::as_sexp(aic)});
  
  return result;
}

/* roxygen
@title Predictive DFM
@description Uses factors from covariates X to predict target variable y. This is similar to FA-VAR but uses DFM factor
  extraction
@export
*/
[[cpp4r::register]] list dfm_predict_model(const doubles_matrix<>& y, const doubles_matrix<>& x,
                                           int n_factors, int p_y = 1, int p_f = 0,
                                           bool include_const = true, int max_iter = 25, 
                                           double tol = 1e-4, int forecast_h = 0) {
  
  Mat<double> Y = as_Mat(y);
  Mat<double> X = as_Mat(x);
  
  int T = Y.n_rows;
  int K = Y.n_cols;
  int N = X.n_rows;
  int P = X.n_cols;
  
  if (T != N) {
    throw std::runtime_error("y and X must have the same number of observations");
  }
  
  if (T <= std::max(p_y, p_f)) {
    throw std::runtime_error("Sample size must be greater than maximum lag order");
  }
  
  if (n_factors >= P) {
    throw std::runtime_error("Number of factors must be less than number of X variables");
  }
  
  // Step 1: Extract factors from covariates X using DFM approach
  auto factor_result = dfm_extract_factors_(X, n_factors);
  Mat<double> F = factor_result.first;
  Mat<double> Lambda = factor_result.second;
  
  // Step 2: Estimate factor dynamics (if p_f > 0)
  Mat<double> Phi;
  Mat<double> F_residuals;
  Mat<double> Omega;
  
  if (p_f > 0) {
    Phi = var_estimate_(as_doubles_matrix(F), p_f, include_const);
    auto factor_fitted_resid = var_fitted_resid_(F, Phi, p_f, include_const);
    F_residuals = factor_fitted_resid.second;
    Omega = (F_residuals.t() * F_residuals) / (F_residuals.n_rows - Phi.n_rows);
  } else {
    // No factor dynamics: use mean as constant
    if (include_const) {
      Phi = mean(F, 0).t();
    } else {
      Phi = zeros<Mat<double>>(1, n_factors);
    }
    F_residuals = zeros<Mat<double>>(T - std::max(p_y, p_f), n_factors);
    Omega = eye<Mat<double>>(n_factors, n_factors) * 0.01;
  }
  
  // Step 3: Estimate predictive regression: y_t = B(L)y_{t-1} + C(L)F_t + u_t
  // Reuse FA-VAR augmented VAR function
  auto var_result = favar_augmented_var_(Y, F, p_y, p_f, include_const);
  Mat<double> beta = std::get<0>(var_result);
  Mat<double> C = std::get<1>(var_result);
  
  // Step 4: Compute fitted values and residuals
  auto fitted_resid = favar_fitted_resid_(Y, F, beta, p_y, p_f, include_const);
  Mat<double> y_fitted = fitted_resid.first;
  Mat<double> residuals = fitted_resid.second;
  
  // Compute residual covariance matrix
  Mat<double> Sigma = (residuals.t() * residuals) / (residuals.n_rows - beta.n_rows);
  
  // Compute AIC
  int n_params = beta.n_rows * K;  // Total number of parameters in augmented VAR
  double aic = aic_metric(residuals, n_params);
  
  // Prepare results list
  writable::list result(16);

  result[0] = as_doubles_matrix(beta);
  result[1] = as_doubles_matrix(Phi);
  result[2] = as_doubles_matrix(Lambda);
  result[3] = as_doubles_matrix(F);
  result[4] = as_doubles_matrix(y_fitted);
  result[5] = as_doubles_matrix(residuals);
  result[6] = as_doubles_matrix(F_residuals);
  result[7] = as_doubles_matrix(Sigma);
  result[8] = as_doubles_matrix(Omega);
  result[9] = cpp4r::as_sexp(n_factors);
  result[10] = cpp4r::as_sexp(p_y);
  result[11] = cpp4r::as_sexp(p_f);
  result[12] = cpp4r::as_sexp(K);
  result[13] = cpp4r::as_sexp(P);
  result[14] = cpp4r::as_sexp(T - std::max(p_y, p_f));
  result[15] = cpp4r::as_sexp(include_const);

  result.names() = {"coefficients", "factor_coefficients", "factor_loadings", "factors",
                    "fitted_values", "residuals", "factor_residuals", "sigma", "omega",
                    "n_factors", "lag_order_y", "lag_order_f", "n_variables", "n_covariates",
                    "n_obs", "include_const"};
  
  // Add forecasts if requested
  if (forecast_h > 0) {
    Mat<double> forecasts = favar_forecast_(Y, F, beta, Phi, p_y, p_f, forecast_h, include_const);
    result.push_back({"forecasts"_nm = as_doubles_matrix(forecasts)});
    result.push_back({"forecast_horizon"_nm = cpp4r::as_sexp(forecast_h)});
  }

  // Add model metrics
  result.push_back({"rmsfe"_nm = cpp4r::as_sexp(rmsfe(Y.rows(std::max(p_y, p_f), T-1), y_fitted))});
  result.push_back({"mae"_nm = cpp4r::as_sexp(mae(Y.rows(std::max(p_y, p_f), T-1), y_fitted))});
  result.push_back({"aic"_nm = cpp4r::as_sexp(aic)});
  
  return result;
}
