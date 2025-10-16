/* Root Mean Squared Forecast Error (RMSFE) */

double rmsfe(const Mat<double>& actual, const Mat<double>& forecast) {
  Mat<double> errors = actual - forecast;
  Mat<double> squared_errors = square(errors);
  
  double mse = accu(squared_errors) / squared_errors.n_elem;
  
  return sqrt(mse);
}

/* Mean Absolute Error (MAE) */

double mae(const Mat<double>& actual, const Mat<double>& forecast) {  
  Mat<double> errors = abs(actual - forecast);
  return accu(errors) / errors.n_elem;
}

/* Akaike Information Criterion (AIC) */

double aic_metric(const Mat<double>& residuals, int n_params) {
  int T = residuals.n_rows;
  
  // Sum of squared residuals
  double ssr = accu(square(residuals));
  
  // AIC = log(SSR/T) + (p+1) * 2/T
  // For multivariate models, use total SSR across all equations
  double aic = log(ssr / T) + n_params * 2.0 / T;
  
  return aic;
}
