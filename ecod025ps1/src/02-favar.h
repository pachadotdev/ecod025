/*
==============================================
Factor-Augmented VAR (FA-VAR)
==============================================
*/

// Extract factors using PCA from large dataset
std::pair<Mat<double>, Mat<double>> favar_extract_factors_(const Mat<double>& X, int n_factors) {  
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

// Estimate factor VAR: F_t = Phi_1*F_{t-1} + ... + Phi_p*F_{t-p} + v_t
Mat<double> favar_factor_var_(const Mat<double>& F, int p_f, bool include_const = true) {
  int N = F.n_rows;
  int n_factors = F.n_cols;

  // Handle p_f = 0 case: no factor dynamics, just return constant term
  if (p_f == 0) {
    if (include_const) {
      // Return mean of factors as constant
      return mean(F, 0).t();  // Column vector of means
    } else {
      // Return zeros
      return zeros<Mat<double>>(1, n_factors);
    }
  }

  if (N <= p_f) {
    throw std::runtime_error("Sample size must be greater than factor lag order p_f");
  }
  
  // Create lagged factors
  Mat<double> F_lag = create_lags_(F, p_f);
  
  // Dependent variable (after losing p_f observations)
  Mat<double> F_dep = F.rows(p_f, F.n_rows-1);
  
  // Add constant if requested
  Mat<double> X;
  if (include_const) {
    Mat<double> ones(F_lag.n_rows, 1, fill::ones);
    X = join_horiz(ones, F_lag);
  } else {
    X = F_lag;
  }
  
  // Estimate factor VAR coefficients: Phi = (X'X)^(-1)(X'F)
  Mat<double> XtX = X.t() * X;
  Mat<double> XtX_inv = inv(XtX);
  Mat<double> Phi = XtX_inv * X.t() * F_dep;
  
  return Phi;
}

// Estimate augmented VAR: y_t = B(L)*y_{t-1} + C(L)*F_t + u_t
std::tuple<Mat<double>, Mat<double>> favar_augmented_var_(
    const Mat<double>& y, const Mat<double>& F, int p_y, int p_f, bool include_const = true) {
  
  int T = y.n_rows;
  int K = y.n_cols;
  
  int max_p = std::max(p_y, p_f);
  
  if (T <= max_p) {
    throw std::runtime_error("Sample size must be greater than maximum lag order");
  }
  
  // Create lagged y variables
  Mat<double> y_lag;
  if (p_y > 0) {
    y_lag = create_lags_(y, p_y);
    // Align with maximum lag
    if (p_y < max_p) {
      Mat<double> y_lag_aligned = y_lag.rows(max_p - p_y, y_lag.n_rows - 1);
      y_lag = y_lag_aligned;
    }
  }
  
  // Create lagged and contemporaneous factors
  Mat<double> F_regressors;
  if (p_f > 0) {
    // Include lagged factors: F_{t-1}, ..., F_{t-p_f}
    Mat<double> F_lag = create_lags_(F, p_f);
    // Align with maximum lag
    if (p_f < max_p) {
      Mat<double> F_lag_aligned = F_lag.rows(max_p - p_f, F_lag.n_rows - 1);
      F_lag = F_lag_aligned;
    }
    
    // Include contemporaneous factors: F_t
    Mat<double> F_contemp = F.rows(max_p, F.n_rows - 1);
    
    // Combine: [F_t, F_{t-1}, ..., F_{t-p_f}]
    F_regressors = join_horiz(F_contemp, F_lag);
  } else {
    // Only contemporaneous factors
    F_regressors = F.rows(max_p, F.n_rows - 1);
  }
  
  // Dependent variable (after losing max_p observations)
  Mat<double> y_dep = y.rows(max_p, T - 1);
  
  // Combine regressors: [y_{t-1}, ..., y_{t-p_y}, F_t, F_{t-1}, ..., F_{t-p_f}]
  Mat<double> X;
  if (p_y > 0) {
    X = join_horiz(y_lag, F_regressors);
  } else {
    X = F_regressors;
  }
  
  // Add constant if requested
  if (include_const) {
    Mat<double> ones(X.n_rows, 1, fill::ones);
    X = join_horiz(ones, X);
  }
  
  // Estimate coefficients: [B, C] = (X'X)^(-1)(X'y)
  Mat<double> XtX = X.t() * X;
  Mat<double> XtX_inv = inv(XtX);
  Mat<double> beta = XtX_inv * X.t() * y_dep;
  
  // Split coefficients into B (for y lags) and C (for factors)
  Mat<double> B, C;
  int start_idx = include_const ? 1 : 0;
  
  if (p_y > 0) {
    B = beta.rows(start_idx, start_idx + K * p_y - 1);
    C = beta.rows(start_idx + K * p_y, beta.n_rows - 1);
  } else {
    B = zeros<Mat<double>>(0, K);  // No y lags
    C = beta.rows(start_idx, beta.n_rows - 1);
  }
  
  return std::make_tuple(beta, C);  // Return full coefficient matrix and factor coefficients
}

// Compute FAVAR fitted values and residuals
std::pair<Mat<double>, Mat<double>> favar_fitted_resid_(
    const Mat<double>& y, const Mat<double>& F, const Mat<double>& beta,
    int p_y, int p_f, bool include_const = true) {
  
  int T = y.n_rows;
  int max_p = std::max(p_y, p_f);
  
  // Create same regressor matrix as in estimation
  Mat<double> y_lag;
  if (p_y > 0) {
    y_lag = create_lags_(y, p_y);
    if (p_y < max_p) {
      y_lag = y_lag.rows(max_p - p_y, y_lag.n_rows - 1);
    }
  }
  
  Mat<double> F_regressors;
  if (p_f > 0) {
    Mat<double> F_lag = create_lags_(F, p_f);
    if (p_f < max_p) {
      F_lag = F_lag.rows(max_p - p_f, F_lag.n_rows - 1);
    }
    Mat<double> F_contemp = F.rows(max_p, F.n_rows - 1);
    F_regressors = join_horiz(F_contemp, F_lag);
  } else {
    F_regressors = F.rows(max_p, F.n_rows - 1);
  }
  
  Mat<double> X;
  if (p_y > 0) {
    X = join_horiz(y_lag, F_regressors);
  } else {
    X = F_regressors;
  }
  
  if (include_const) {
    Mat<double> ones(X.n_rows, 1, fill::ones);
    X = join_horiz(ones, X);
  }
  
  // Dependent variable
  Mat<double> y_dep = y.rows(max_p, T - 1);
  
  // Compute fitted values and residuals
  Mat<double> y_fitted = X * beta;
  Mat<double> residuals = y_dep - y_fitted;
  
  return std::make_pair(y_fitted, residuals);
}

// FAVAR forecasting
Mat<double> favar_forecast_(const Mat<double>& y, const Mat<double>& F,
                           const Mat<double>& beta_y, const Mat<double>& Phi_f,
                           int p_y, int p_f, int h, bool include_const = true) {
  
  int T = y.n_rows;
  int K = y.n_cols;
  int n_factors = F.n_cols;

  int Phi_r = Phi_f.n_rows;
  int Phi_c = Phi_f.n_cols;
  
  // Initialize forecasts
  Mat<double> y_forecast(h, K);
  Mat<double> F_forecast(h, n_factors);
  
  // Get initial conditions
  Mat<double> y_init = y.rows(T - std::max(p_y, 1), T - 1);
  
  // Handle factor forecasting based on p_f
  if (p_f == 0) {
    // No factor dynamics: use constant (mean or zero)
    for (int i = 0; i < h; ++i) {
      if (Phi_r == 1 && Phi_c == n_factors) {
        F_forecast.row(i) = Phi_f;  // Constant term
      } else {
        F_forecast.row(i) = zeros<Mat<double>>(1, n_factors);
      }
    }
  } else {
    // Forecast factors using factor VAR
    Mat<double> F_init = F.rows(T - std::max(p_f, 1), T - 1);
    Mat<double> F_current = vectorise(F_init.t()).t();
    
    for (int i = 0; i < h; ++i) {
      Mat<double> X_f;
      if (include_const) {
        Mat<double> ones(1, 1, fill::ones);
        X_f = join_horiz(ones, F_current);
      } else {
        X_f = F_current;
      }
      
      Mat<double> F_pred = X_f * Phi_f;
      F_forecast.row(i) = F_pred;
      
      // Update for next iteration
      if (i < h - 1 && p_f > 1) {
        Mat<double> new_F_current = join_horiz(F_pred, F_current.cols(0, n_factors * (p_f - 1) - 1));
        F_current = new_F_current;
      } else if (p_f == 1) {
        F_current = F_pred;
      }
    }
  }
  
  // Now forecast y using factor forecasts
  Mat<double> y_current = vectorise(y_init.t()).t();
  Mat<double> F_extended = join_vert(F, F_forecast);
  
  for (int i = 0; i < h; ++i) {
    Mat<double> X_y;
    
    // Add y lags if p_y > 0
    if (p_y > 0) {
      X_y = y_current;
    }
    
    // Add factors (contemporaneous and lagged)
    Mat<double> F_for_pred = F_extended.row(T + i);
    if (p_f > 0) {
      for (int lag = 1; lag <= p_f; ++lag) {
        if (T + i - lag >= 0) {
          Mat<double> F_lag_t = F_extended.row(T + i - lag);
          F_for_pred = join_horiz(F_for_pred, F_lag_t);
        }
      }
    }
    
    if (p_y > 0) {
      X_y = join_horiz(X_y, F_for_pred);
    } else {
      X_y = F_for_pred;
    }
    
    // Add constant
    if (include_const) {
      Mat<double> ones(1, 1, fill::ones);
      X_y = join_horiz(ones, X_y);
    }
    
    Mat<double> y_pred = X_y * beta_y;
    y_forecast.row(i) = y_pred;
    
    // Update y_current for next iteration
    if (i < h - 1 && p_y > 1) {
      Mat<double> new_y_current = join_horiz(y_pred, y_current.cols(0, K * (p_y - 1) - 1));
      y_current = new_y_current;
    } else if (p_y == 1) {
      y_current = y_pred;
    }
  }
  
  return y_forecast;
}

/* roxygen
@title Main FAVAR estimation function
@export
*/
[[cpp4r::register]] list favar_model(const doubles_matrix<>& y, const doubles_matrix<>& x,
                                     int n_factors, int p_y = 1, int p_f = 1,
                                     bool include_const = true, int forecast_h = 0) {
  
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
  
  // Step 1: Extract factors from large dataset X
  auto factor_result = favar_extract_factors_(X, n_factors);
  Mat<double> F = factor_result.first;
  Mat<double> Lambda = factor_result.second;
  
  // Step 2: Estimate factor VAR
  Mat<double> Phi = favar_factor_var_(F, p_f, include_const);
  
  // Step 3: Estimate augmented VAR
  auto var_result = favar_augmented_var_(Y, F, p_y, p_f, include_const);
  Mat<double> beta = std::get<0>(var_result);
  Mat<double> C = std::get<1>(var_result);
  
  // Step 4: Compute fitted values and residuals
  auto fitted_resid = favar_fitted_resid_(Y, F, beta, p_y, p_f, include_const);
  Mat<double> y_fitted = fitted_resid.first;
  Mat<double> residuals = fitted_resid.second;
  
  // Compute residual covariance matrix
  Mat<double> Sigma = (residuals.t() * residuals) / (residuals.n_rows - beta.n_rows);
  
  // Compute factor residuals
  auto factor_fitted_resid = var_fitted_resid_(F, Phi, p_f, include_const);
  Mat<double> F_residuals = factor_fitted_resid.second;
  Mat<double> Omega = (F_residuals.t() * F_residuals) / (F_residuals.n_rows - Phi.n_rows);
  
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
  result[13] = cpp4r::as_sexp(N);
  result[14] = cpp4r::as_sexp(T - std::max(p_y, p_f));
  result[15] = cpp4r::as_sexp(include_const);

  result.names() = {"coefficients", "factor_coefficients", "factor_loadings", "factors",
                    "fitted_values", "residuals", "factor_residuals", "sigma", "omega",
                    "n_factors", "lag_order_y", "lag_order_f", "n_variables", "n_observables",
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

