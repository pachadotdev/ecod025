// Bayesian VAR model with Minnesota prior for forecasting
data {
  int<lower=1> T;           // Number of time periods
  int<lower=1> K;           // Number of variables (3: CPI, GDP deflator, GDP growth)
  int<lower=1> p;           // Number of lags
  int<lower=1> h;           // Forecast horizon
  matrix[T, K] Y;           // Observed data
  
  // Minnesota prior hyperparameters
  real<lower=0> lambda;     // Overall tightness
  real<lower=0> tau;        // Relative tightness of own lags
  real<lower=0> epsilon;    // Small constant for numerical stability
}

transformed data {
  int T_eff = T - p;        // Effective sample size
  matrix[T_eff, K*p] X;    // Lagged variables matrix
  matrix[T_eff, K] Y_eff;  // Dependent variables
  
  // Construct lagged matrix
  for (t in 1:T_eff) {
    Y_eff[t] = Y[t + p];
    for (lag in 1:p) {
      for (k in 1:K) {
        X[t, (lag-1)*K + k] = Y[t + p - lag, k];
      }
    }
  }
}

parameters {
  matrix[K*p, K] B;         // VAR coefficients
  vector[K] c;              // Intercepts
  cholesky_factor_corr[K] L_Omega;  // Cholesky factor of correlation matrix
  vector<lower=0>[K] sigma;         // Standard deviations
}

transformed parameters {
  matrix[K, K] Sigma;      // Covariance matrix
  matrix[K, K] L_Sigma;    // Cholesky factor of Sigma
  L_Sigma = diag_pre_multiply(sigma, L_Omega);
  Sigma = L_Sigma * L_Sigma';
}

model {
  matrix[T_eff, K] mu;
  
  // Priors for error structure (put these FIRST to establish sigma values)
  sigma ~ normal(1, 1);  // More informative prior centered at 1
  L_Omega ~ lkj_corr_cholesky(2.0);
  
  // Minnesota prior on VAR coefficients (now sigma is well-defined)
  for (k in 1:K) {
    c[k] ~ normal(0, 10);  // Weakly informative prior on intercepts
    
    for (j in 1:K) {
      for (lag in 1:p) {
        int idx = (lag-1)*K + j;
        real prior_mean = (k == j && lag == 1) ? 0.8 : 0.0;  // AR(1) coefficient = 0.8 for own first lag
        real prior_sd;
        
        if (k == j) {
          // Own lags: simple shrinkage
          prior_sd = lambda / pow(lag, 2);
        } else {
          // Cross lags: add epsilon to prevent division by zero
          prior_sd = lambda * tau / (pow(lag, 2) * (sigma[j] + epsilon));
        }
        
        B[idx, k] ~ normal(prior_mean, prior_sd);
      }
    }
  }
  
  // Likelihood
  mu = X * B;
  for (i in 1:K) {
    mu[, i] = mu[, i] + c[i];
  }
  for (t in 1:T_eff) {
    Y_eff[t] ~ multi_normal_cholesky(to_vector(mu[t]), L_Sigma);
  }
}

generated quantities {
  matrix[h, K] Y_forecast;           // Forecasts
  matrix[h, K] Y_forecast_mean;      // Mean forecasts
  matrix[T_eff, K] fitted;           // Fitted values
  matrix[T_eff, K] residuals;        // Residuals
  array[T_eff] real log_lik;         // Log likelihood for LOO
  
  // Compute fitted values and residuals
  fitted = X * B;
  for (i in 1:K) {
    fitted[, i] = fitted[, i] + c[i];
  }
  residuals = Y_eff - fitted;
  
  // Compute log likelihood
  for (t in 1:T_eff) {
    log_lik[t] = multi_normal_lpdf(Y_eff[t] | fitted[t], Sigma);
  }
  
  // Generate forecasts
  {
    matrix[p, K] Y_last;  // Last p observations
    matrix[h + p, K] Y_extended;  // Extended series for forecasting
    
    // Initialize with last p observations
    for (i in 1:p) {
      Y_last[i] = Y[T - p + i];
      Y_extended[i] = Y_last[i];
    }
    
    // Generate h-step ahead forecasts
    for (horizon in 1:h) {
      vector[K*p] x_forecast;
      
      // Construct lagged variables for forecasting
      for (lag in 1:p) {
        for (k in 1:K) {
          if (horizon - lag <= 0) {
            // Use actual data
            x_forecast[(lag-1)*K + k] = Y_last[p + horizon - lag, k];
          } else {
            // Use previous forecasts
            x_forecast[(lag-1)*K + k] = Y_extended[p + horizon - lag, k];
          }
        }
      }
      
      // Mean forecast
      Y_forecast_mean[horizon] = (B' * x_forecast + c)';
      
      // Forecast with uncertainty
      Y_forecast[horizon] = multi_normal_cholesky_rng(
        B' * x_forecast + c, 
        L_Sigma
      )';
      
      // Store for next iteration
      Y_extended[p + horizon] = Y_forecast[horizon];
    }
  }
}
