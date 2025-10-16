// Factor-Augmented VAR (FA-VAR)

// Extract factors using PCA from time series data
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
Mat<double> favar_factor_var_(const Mat<double>& F, int p, bool include_const = false) {
  int T = F.n_rows;

  if (p == 0 || T <= p) {
    throw std::runtime_error("Invalid lag order for factor VAR");
  }
  
  // Use existing VAR estimation function
  Mat<double> Phi = var_estimate_(as_doubles_matrix(F), p, include_const);
  
  return Phi;
}

// Estimate augmented VAR: y_t = B(L)*y_{t-1} + C(L)*F_{t-1} + u_t
// Uses only LAGGED factors to avoid data leakage
std::pair<Mat<double>, Mat<double>> favar_augmented_var_(
    const Mat<double>& y, const Mat<double>& F, int p_y, int p_f, bool include_const = false) {
  
  int T = y.n_rows;
  int K = y.n_cols;
  
  int max_p = std::max(p_y, p_f);
  
  if (T <= max_p) {
    throw std::runtime_error("Sample size must be greater than lag order");
  }
  
  // Create lagged y variables
  Mat<double> y_lag;
  Mat<double> y_dep = y.rows(max_p, T - 1);
  
  if (p_y > 0) {
    y_lag = create_lags_(y, p_y);
    // Align if needed
    if (p_y < max_p) {
      y_lag = y_lag.rows(max_p - p_y, y_lag.n_rows - 1);
    }
  }
  
  // Create factor regressors: [F_{t-1}, ..., F_{t-p_f}] (only lagged factors)
  Mat<double> F_regressors;
  
  if (p_f > 0) {
    Mat<double> F_lag = create_lags_(F, p_f);
    // Align if needed
    if (p_f < max_p) {
      F_lag = F_lag.rows(max_p - p_f, F_lag.n_rows - 1);
    }
    F_regressors = F_lag;
  } else {
    throw std::runtime_error("p_f must be at least 1 for FAVAR");
  }
  
  // Combine regressors: [y_{t-1}, ..., y_{t-p_y}, F_{t-1}, ..., F_{t-p_f}]
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
  Mat<double> beta = solve(X.t() * X, X.t() * y_dep);
  
  // Extract factor coefficients C
  int start_idx = include_const ? 1 : 0;
  int factor_start = start_idx + (p_y > 0 ? K * p_y : 0);
  Mat<double> C = beta.rows(factor_start, beta.n_rows - 1);
  
  return std::make_pair(beta, C);
}

// Compute FAVAR fitted values and residuals
std::pair<Mat<double>, Mat<double>> favar_fitted_resid_(
    const Mat<double>& y, const Mat<double>& F, const Mat<double>& beta,
    int p_y, int p_f, bool include_const = false) {
  
  int T = y.n_rows;
  int max_p = std::max(p_y, p_f);
  
  // Reconstruct regressor matrix
  Mat<double> y_dep = y.rows(max_p, T - 1);
  Mat<double> X;
  
  if (p_y > 0) {
    Mat<double> y_lag = create_lags_(y, p_y);
    if (p_y < max_p) {
      y_lag = y_lag.rows(max_p - p_y, y_lag.n_rows - 1);
    }
    
    Mat<double> F_regressors;
    
    if (p_f > 0) {
      Mat<double> F_lag = create_lags_(F, p_f);
      if (p_f < max_p) {
        F_lag = F_lag.rows(max_p - p_f, F_lag.n_rows - 1);
      }
      F_regressors = F_lag;
    }
    
    X = join_horiz(y_lag, F_regressors);
  } else {
    Mat<double> F_regressors;
    
    if (p_f > 0) {
      Mat<double> F_lag = create_lags_(F, p_f);
      if (p_f < max_p) {
        F_lag = F_lag.rows(max_p - p_f, F_lag.n_rows - 1);
      }
      F_regressors = F_lag;
    }
    
    X = F_regressors;
  }
  
  if (include_const) {
    Mat<double> ones(X.n_rows, 1, fill::ones);
    X = join_horiz(ones, X);
  }
  
  // Compute fitted values and residuals
  Mat<double> y_fitted = X * beta;
  Mat<double> residuals = y_dep - y_fitted;
  
  return std::make_pair(y_fitted, residuals);
}

// FAVAR forecasting with lagged factors only
Mat<double> favar_forecast_(const Mat<double>& y, const Mat<double>& F,
                           const Mat<double>& beta_y, const Mat<double>& Phi_f,
                           int p_y, int p_f, int h, bool include_const = false) {
  
  int T = y.n_rows;
  int K = y.n_cols;
  int n_factors = F.n_cols;
  
  // Initialize forecasts
  Mat<double> y_forecast(h, K);
  Mat<double> F_forecast(h, n_factors);
  
  // Step 1: Forecast factors using factor VAR
  Mat<double> F_init = F.rows(T - p_f, T - 1);
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
    
    // Update state for next iteration
    if (i < h - 1 && p_f > 1) {
      F_current = join_horiz(F_pred, F_current.cols(0, n_factors * (p_f - 1) - 1));
    } else if (p_f == 1) {
      F_current = F_pred;
    }
  }
  
  // Step 2: Forecast y using forecasted factors (lagged factors only)
  Mat<double> y_init = y.rows(T - std::max(p_y, 1), T - 1);
  Mat<double> y_current = (p_y > 0) ? vectorise(y_init.t()).t() : Mat<double>();
  
  // Extend F with forecasts for easier indexing
  Mat<double> F_extended = join_vert(F, F_forecast);
  
  for (int i = 0; i < h; ++i) {
    Mat<double> X_y;
    
    // Add y lags if p_y > 0
    if (p_y > 0) {
      X_y = y_current;
    }
    
    // Add lagged factors: F_{T+i-1}, ..., F_{T+i-p_f} (no contemporaneous factor)
    Mat<double> F_current_forecast;
    if (p_f > 0) {
      for (int lag = 1; lag <= p_f; ++lag) {
        if (T + i - lag >= 0) {
          Mat<double> F_lag_t = F_extended.row(T + i - lag);
          if (lag == 1) {
            F_current_forecast = F_lag_t;
          } else {
            F_current_forecast = join_horiz(F_current_forecast, F_lag_t);
          }
        }
      }
    }
    
    // Combine y lags and factors
    if (p_y > 0) {
      X_y = join_horiz(X_y, F_current_forecast);
    } else {
      X_y = F_current_forecast;
    }
    
    // Add constant if needed
    if (include_const) {
      Mat<double> ones(1, 1, fill::ones);
      X_y = join_horiz(ones, X_y);
    }
    
    Mat<double> y_pred = X_y * beta_y;
    y_forecast.row(i) = y_pred;
    
    // Update y_current for next iteration
    if (p_y > 1 && i < h - 1) {
      y_current = join_horiz(y_pred, y_current.cols(0, K * (p_y - 1) - 1));
    } else if (p_y == 1) {
      y_current = y_pred;
    }
  }
  
  return y_forecast;
}

/* roxygen
@title Factor-Augmented VAR (FA-VAR) Model
@description Estimates a FA-VAR model using PCA for factor extraction.
@param y Time series vector (T x 1)
@param n_lags Number of lags of y to include in X for factor extraction
@param n_factors Number of latent factors to extract (r < n_lags + 1)
@param p_y VAR lag order for y dynamics
@param p_f VAR lag order for factor dynamics
@param include_const Include constant term in VARs
@param forecast_h Forecast horizon (0 for no forecast)
@export
*/
[[cpp4r::register]] list favar_model(const doubles_matrix<>& y, int n_lags,
                                     int n_factors, int p_y = 1, int p_f = 1,
                                     bool include_const = false, int forecast_h = 0) {
  
  Mat<double> Y = as_Mat(y);
  int T = Y.n_rows;
  int K = Y.n_cols;
  
  if (K != 1) {
    throw std::runtime_error("FA-VAR requires univariate y (single column)");
  }
  
  if (T <= n_lags + std::max(p_y, p_f)) {
    throw std::runtime_error("Insufficient observations for lags");
  }
  
  // Step 1: Create X matrix [y_t, y_{t-1}, ..., y_{t-n_lags}]
  Mat<double> X_lags = create_lags_(Y, n_lags);
  Mat<double> Y_contemp = Y.rows(n_lags, T - 1);
  Mat<double> X = join_horiz(Y_contemp, X_lags);
  
  int N = X.n_cols;  // n_lags + 1
  
  if (n_factors >= N) {
    throw std::runtime_error("Number of factors must be less than number of variables in X");
  }
  
  // Step 2: Extract factors from X using PCA
  auto factor_result = favar_extract_factors_(X, n_factors);
  Mat<double> F = factor_result.first;
  Mat<double> Lambda = factor_result.second;
  
  // Step 3: Estimate factor VAR
  Mat<double> Phi = favar_factor_var_(F, p_f, include_const);
  
  // Step 4: Estimate augmented VAR (y on its lags + lagged factors only)
  // Need to align y with F (both start at observation n_lags)
  Mat<double> Y_aligned = Y.rows(n_lags, T - 1);
  
  auto var_result = favar_augmented_var_(Y_aligned, F, p_y, p_f, include_const);
  Mat<double> beta = var_result.first;
  Mat<double> C = var_result.second;
  
  // Step 5: Compute fitted values and residuals
  auto fitted_resid = favar_fitted_resid_(Y_aligned, F, beta, p_y, p_f, include_const);
  Mat<double> y_fitted = fitted_resid.first;
  Mat<double> residuals = fitted_resid.second;
  
  // Compute residual covariance
  Mat<double> Sigma = (residuals.t() * residuals) / (residuals.n_rows - beta.n_rows);
  
  // Compute factor residuals
  auto factor_fitted_resid = var_fitted_resid_(F, Phi, p_f, include_const);
  Mat<double> F_residuals = factor_fitted_resid.second;
  Mat<double> Omega = (F_residuals.t() * F_residuals) / (F_residuals.n_rows - Phi.n_rows);
  
  // Compute AIC
  int n_params = beta.n_rows * K;
  double aic = aic_metric(residuals, n_params);
  
  // Prepare results list
  writable::list result;
  
  result.push_back({"coefficients"_nm = as_doubles_matrix(beta)});
  result.push_back({"factor_coefficients"_nm = as_doubles_matrix(Phi)});
  result.push_back({"factor_loadings"_nm = as_doubles_matrix(Lambda)});
  result.push_back({"factors"_nm = as_doubles_matrix(F)});
  result.push_back({"fitted_values"_nm = as_doubles_matrix(y_fitted)});

  result.push_back({"residuals"_nm = as_doubles_matrix(residuals)});
  result.push_back({"factor_residuals"_nm = as_doubles_matrix(F_residuals)});
  result.push_back({"sigma"_nm = as_doubles_matrix(Sigma)});
  result.push_back({"omega"_nm = as_doubles_matrix(Omega)});
  result.push_back({"n_factors"_nm = cpp4r::as_sexp(n_factors)});

  result.push_back({"n_lags"_nm = cpp4r::as_sexp(n_lags)});
  result.push_back({"lag_order_y"_nm = cpp4r::as_sexp(p_y)});
  result.push_back({"lag_order_f"_nm = cpp4r::as_sexp(p_f)});
  result.push_back({"n_obs"_nm = cpp4r::as_sexp(y_fitted.n_rows)});
  result.push_back({"include_const"_nm = cpp4r::as_sexp(include_const)});

  // Compute RMSFE and MAE using helper functions
  // Need to align Y_aligned with fitted values (both start after max_p lags)
  int max_p = std::max(p_y, p_f);
  Mat<double> Y_actual = Y_aligned.rows(max_p, Y_aligned.n_rows - 1);
  
  // result.push_back({"observed_values"_nm = as_doubles_matrix(Y_actual)});
  result.push_back({"rmsfe"_nm = cpp4r::as_sexp(rmsfe(Y_actual, y_fitted))});
  result.push_back({"mae"_nm = cpp4r::as_sexp(mae(Y_actual, y_fitted))});
  result.push_back({"aic"_nm = cpp4r::as_sexp(aic)});
  
  // Add forecasts if requested
  if (forecast_h > 0) {
    Mat<double> forecasts = favar_forecast_(Y_aligned, F, beta, Phi, p_y, p_f, forecast_h, include_const);
    result.push_back({"forecasts"_nm = as_doubles_matrix(forecasts)});
    result.push_back({"forecast_horizon"_nm = cpp4r::as_sexp(forecast_h)});
  }
  
  return result;
}
