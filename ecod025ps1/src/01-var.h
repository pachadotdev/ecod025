/*
==============================================
Vector Autoregressions (VAR)
===========================================
*/

// Create lagged matrix for VAR model
Mat<double> create_lags_(const Mat<double>& Y, int p) {
  int T = Y.n_rows;
  int K = Y.n_cols;
  
  // Effective sample size after lagging
  int T_eff = T - p;
  
  // Create lagged matrix: [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
  Mat<double> X_lag(T_eff, K * p);
  
  for (int lag = 1; lag <= p; ++lag) {
    for (int t = 0; t < T_eff; ++t) {
      X_lag.submat(t, (lag-1)*K, t, lag*K-1) = Y.row(t + p - lag);
    }
  }
  
  return X_lag;
}

// Estimate VAR model using equation-by-equation OLS
Mat<double> var_estimate_(const doubles_matrix<>& y, int p, bool include_const = true) {
  Mat<double> Y = as_Mat(y);
  int T = Y.n_rows;
  
  if (T <= p) {
    throw std::runtime_error("Sample size must be greater than lag order p");
  }
  
  // Create lagged regressors
  Mat<double> Y_lag = create_lags_(Y, p);
  
  // Dependent variable (after losing p observations)
  Mat<double> Y_dep = Y.rows(p, T-1);
  
  // Add constant if requested
  Mat<double> X;
  if (include_const) {
    Mat<double> ones(Y_lag.n_rows, 1, fill::ones);
    X = join_horiz(ones, Y_lag);
  } else {
    X = Y_lag;
  }
  
  // Estimate VAR coefficients: A = (X'X)^(-1)(X'Y)
  Mat<double> XtX = X.t() * X;
  Mat<double> XtX_inv = inv(XtX);
  Mat<double> A = XtX_inv * X.t() * Y_dep;
  
  return A;
}

// Compute VAR residuals and fitted values
std::pair<Mat<double>, Mat<double>> var_fitted_resid_(const Mat<double>& Y, const Mat<double>& A, 
                                                      int p, bool include_const = true) {
  int T = Y.n_rows;
  
  // Create lagged regressors
  Mat<double> Y_lag = create_lags_(Y, p);
  
  // Dependent variable (after losing p observations)
  Mat<double> Y_dep = Y.rows(p, T-1);
  
  // Add constant if requested
  Mat<double> X;
  if (include_const) {
    Mat<double> ones(Y_lag.n_rows, 1, fill::ones);
    X = join_horiz(ones, Y_lag);
  } else {
    X = Y_lag;
  }
  
  // Compute fitted values and residuals
  Mat<double> Y_fitted = X * A;
  Mat<double> residuals = Y_dep - Y_fitted;
  
  return std::make_pair(Y_fitted, residuals);
}

// Create VAR companion matrix for analysis and forecasting
Mat<double> var_companion_(const Mat<double>& A, int p, int K, bool include_const = true) {
  // Extract coefficient matrices (exclude constant if present)
  Mat<double> A_coeff;
  if (include_const) {
    A_coeff = A.rows(1, A.n_rows-1);  // Skip first row (constants)
  } else {
    A_coeff = A;
  }
  
  // Create companion matrix
  Mat<double> F(K*p, K*p, fill::zeros);
  
  // Fill first K rows with coefficient matrices A1, A2, ..., Ap
  F.submat(0, 0, K-1, K*p-1) = A_coeff.t();
  
  // Fill identity blocks for lagged terms
  if (p > 1) {
    Mat<double> I_K = eye<Mat<double>>(K, K);
    for (int i = 1; i < p; ++i) {
      F.submat(i*K, (i-1)*K, (i+1)*K-1, i*K-1) = I_K;
    }
  }
  
  return F;
}

// VAR forecasting function
Mat<double> var_forecast_(const Mat<double>& Y, const Mat<double>& A, 
                         int p, int h, bool include_const = true) {
  int T = Y.n_rows;
  int K = Y.n_cols;
  
  // Get last p observations for initialization
  Mat<double> Y_init = Y.rows(T-p, T-1);
  
  // Initialize forecast container
  Mat<double> forecasts(h, K);
  
  // Current state vector (flatten last p observations)
  Mat<double> y_current = vectorise(Y_init.t()).t();  // Reshape to row vector
  
  // Extract coefficients
  Mat<double> const_term;
  Mat<double> A_coeff;
  
  if (include_const) {
    const_term = A.row(0);
    A_coeff = A.rows(1, A.n_rows-1);
  } else {
    const_term = zeros<Mat<double>>(1, K);
    A_coeff = A;
  }
  
  // Create lagged design matrix for current state
  Mat<double> X_lag = y_current;
  
  // Forecast h periods ahead
  for (int i = 0; i < h; ++i) {
    // Forecast: y_{T+i+1} = c + A1*y_{T+i} + ... + Ap*y_{T+i-p+1}
    Mat<double> y_forecast = const_term + X_lag * A_coeff;
    forecasts.row(i) = y_forecast;
    
    // Update lagged matrix for next forecast
    if (i < h-1) {
      Mat<double> new_X_lag = join_horiz(y_forecast, X_lag.cols(0, K*(p-1)-1));
      X_lag = new_X_lag;
    }
  }
  
  return forecasts;
}
/* roxygen
@title Main VAR estimation function that returns everything
@export
*/
[[cpp4r::register]] list var_model(const doubles_matrix<>& y, int p, 
                                   bool include_const = true, int forecast_h = 0) {
  Mat<double> Y = as_Mat(y);
  int T = Y.n_rows;
  int K = Y.n_cols;
  
  if (T <= p) {
    throw std::runtime_error("Sample size must be greater than lag order p");
  }
  
  // Estimate VAR coefficients
  Mat<double> A = var_estimate_(y, p, include_const);
  
  // Compute fitted values and residuals
  auto fitted_resid = var_fitted_resid_(Y, A, p, include_const);
  Mat<double> Y_fitted = fitted_resid.first;
  Mat<double> residuals = fitted_resid.second;
  
  // Compute residual covariance matrix
  Mat<double> Sigma = (residuals.t() * residuals) / (residuals.n_rows - A.n_rows);
  
  // Create companion matrix
  Mat<double> F = var_companion_(A, p, K, include_const);
  
  // Compute AIC
  int n_params = A.n_rows * K;  // Total number of parameters
  double aic = aic_metric(residuals, n_params);
  
  // Prepare results list
  writable::list result(9);

  result[0] = as_doubles_matrix(A);
  result[1] = as_doubles_matrix(Y_fitted);
  result[2] = as_doubles_matrix(residuals);
  result[3] = as_doubles_matrix(Sigma);
  result[4] = as_doubles_matrix(F);
  result[5] = cpp4r::as_sexp(p);
  result[6] = cpp4r::as_sexp(K);
  result[7] = cpp4r::as_sexp(T - p);  // Effective sample size
  result[8] = cpp4r::as_sexp(include_const);

  result.names() = {"coefficients", "fitted_values", "residuals", "sigma", 
                         "companion_matrix", "lag_order", "n_variables", 
                         "n_obs", "include_const"};
  
  // Add forecasts if requested
  if (forecast_h > 0) {
    Mat<double> forecasts = var_forecast_(Y, A, p, forecast_h, include_const);
    result.push_back({"forecasts"_nm = as_doubles_matrix(forecasts)});
    result.push_back({"forecast_horizon"_nm = cpp4r::as_sexp(forecast_h)});
  }

  // Add model metrics
  result.push_back({"rmsfe"_nm = cpp4r::as_sexp(rmsfe(Y.rows(p, T-1), Y_fitted))});
  result.push_back({"mae"_nm = cpp4r::as_sexp(mae(Y.rows(p, T-1), Y_fitted))});
  result.push_back({"aic"_nm = cpp4r::as_sexp(aic)});
  
  return result;
}
