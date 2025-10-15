/*
==============================================
Root Mean Squared Forecast Error (RMSFE)
===========================================
*/

double rmsfe(const Mat<double>& actual, const Mat<double>& forecast) {
  // if (actual.n_rows != forecast.n_rows || actual.n_cols != forecast.n_cols) {
  //   throw std::runtime_error("Actual and forecast matrices must have the same dimensions");
  // }
  
  Mat<double> errors = actual - forecast;
  Mat<double> squared_errors = square(errors);
  Mat<double> mean_squared_errors = mean(squared_errors, 0); // Mean across rows for each variable
  Mat<double> rmsfe_values = sqrt(mean_squared_errors);
  
  // Return average RMSFE across all variables
  return as_scalar(mean(rmsfe_values));
}

/*
==============================================
Mean Absolute Error (MAE)
==============================================
*/

double mae(const Mat<double>& actual, const Mat<double>& forecast) {
  // if (actual.n_rows != forecast.n_rows || actual.n_cols != forecast.n_cols) {
  //   throw std::runtime_error("Actual and forecast matrices must have the same dimensions");
  // }
  
  Mat<double> errors = abs(actual - forecast);
  Mat<double> mean_absolute_errors = mean(errors, 0); // Mean across rows for each variable
  
  // Return average MAE across all variables
  return as_scalar(mean(mean_absolute_errors));
}

/*
==============================================
Akaike Information Criterion (AIC)
==============================================
*/

double aic_metric(const Mat<double>& residuals, int n_params) {
  int T = residuals.n_rows;
  
  // Sum of squared residuals
  double ssr = accu(square(residuals));
  
  // AIC = log(SSR/T) + (p+1) * 2/T
  // For multivariate models, use total SSR across all equations
  double aic = log(ssr / T) + n_params * 2.0 / T;
  
  return aic;
}
