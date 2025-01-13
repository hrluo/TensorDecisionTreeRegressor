setwd('/media/hrluo/WORK1/tensorTree/data')
#install.packages("TRES")
library(TRES)

library(reticulate)

# Import NumPy through reticulate
numpy <- import("numpy")

## Load data "bat"
data("bat")
x <- bat$x
y <- bat$y

## Fitting with OLS and 1D envelope method.
fit_ols <- TRR.fit(x, y, method="standard")
fit_1D <- TRR.fit(x, y, u = c(14,14), method="1D") # pass envelope rank (14,14)

## Print cofficient
coef(fit_1D)

## Print the summary
summary(fit_1D)

## Extract the mean squared error, p-value and standard error from summary
summary(fit_1D)$mse
summary(fit_1D)$p_val
summary(fit_1D)$se

## Make the prediction on the original dataset
predict(fit_1D, x)

## Draw the plots of two-way coefficient tensor (i.e., matrix) and p-value tensor.
plot(fit_ols)
plot(fit_1D)
 
# Save the numpy arrays to a .npz file
numpy$save("bat_tensor.npy", x)
numpy$save("bat_scalar.npy", y@data)


data("EEG")
x <- EEG$x
y <- EEG$y
## Estimate the envelope dimension, the output should be c(1,1).

u <- TRRdim(x, y)$u
u <- c(1,1)

## Fit the dataset with TRR.fit and draw the coefficient plot and p-value plot
TRRfit_1D <- TRR.fit(x, y, u, method = "1D")
plot(fit_1D, xlab = "Time", ylab = "Channels")
## Uncomment display the plots from different methods.
TRRfit_ols <- TRR.fit(x, y, method = "standard")
TRRfit_FG <- TRR.fit(x, y, u, method = "FG")
TRRfit_ECD <- TRR.fit(x, y, u, method = "ECD")
TRRfit_pls <- TRR.fit(x, y, u, method = "PLS")
plot(fit_ols, xlab = "Time", ylab = "Channels")
plot(fit_pls, xlab = "Time", ylab = "Channels")

# Save the numpy arrays to a .npz file
numpy$save("EEG_tensor.npy", x)
numpy$save("EEG_scalar.npy", y@data)
numpy$save("EEG_TRRfit_1d_coef.npy", TRRfit_1D$coefficients@data)
numpy$save("EEG_TRRfit_ols_coef.npy", TRRfit_ols$coefficients@data)
numpy$save("EEG_TRRfit_FG_coef.npy", TRRfit_FG$coefficients@data)
numpy$save("EEG_TRRfit_ECD_coef.npy", TRRfit_ECD$coefficients@data)
numpy$save("EEG_TRRfit_pls_coef.npy", TRRfit_pls$coefficients@data)

## Fit the dataset with TPR.fit and draw the coefficient plot and p-value plot
TPRfit_1D <- TPR.fit(y, x, u, method = "1D")
plot(fit_1D, xlab = "Time", ylab = "Channels")
## Uncomment display the plots from different methods.
TPRfit_ols <- TPR.fit(y, x, method = "standard")
TPRfit_FG <- TPR.fit(y, x, u, method = "FG")
TPRfit_ECD <- TPR.fit(y, x, u, method = "ECD")
TPRfit_pls <- TPR.fit(y, x, u, method = "PLS")
plot(fit_ols, xlab = "Time", ylab = "Channels")
plot(fit_pls, xlab = "Time", ylab = "Channels")

# Save the numpy arrays to a .npz file
numpy$save("EEG_tensor.npy", x)
numpy$save("EEG_scalar.npy", y@data)
numpy$save("EEG_TPRfit_1d_coef.npy", TPRfit_1D$coefficients@data)
numpy$save("EEG_TPRfit_ols_coef.npy", TPRfit_ols$coefficients@data)
numpy$save("EEG_TPRfit_FG_coef.npy", TPRfit_FG$coefficients@data)
numpy$save("EEG_TPRfit_ECD_coef.npy", TPRfit_ECD$coefficients@data)
numpy$save("EEG_TPRfit_pls_coef.npy", TPRfit_pls$coefficients@data)



 

