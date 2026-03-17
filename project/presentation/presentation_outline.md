# Project Presentation Outline

## Overview of trading strategy
- We implement a trading strategy which uses option data to calculate the volatility surface. From the volatility surface we extract the skew of the surface. We do this for each individual stock and sector etfs to get skew time series. We then treat the skew time series of stocks and the skew time series of the sector etfs as cointegrated. We trade the spread of the cointegrated series and bet on mean reversal. We also delta hedge to avoid exposure to the underlying.

## Data cleaning
- We started with about 10GB of data
- We only use data between 2015-2019
- We filter out dates with less than a week maturity (because these tend to be noisy)
- We filter out data with too high implied volatility (some values exceeded 800% vol)

## Method
In this section we describe what we did and also what we could have done differently if we had more time
For this project we used the simplified setup below
For each day and each ticker:
- Choose one specific time to expiration calculate the implied volatility. 
    - If the specific time to expiration is not available use the two closest and use linear interpolation (justified because the closest time to expriations are quite close so second order effects are small). One could also use more advanced interpolation
    - Instead of using a specific time to expiration you can use the whole surface.
- Only use OTM calls and OTM puts because that's where liquidity is
    - Here we could also weighted by open interest
- Fit a second degree polynomial to the implied volatility, interpret the beta for the linear term to be the skew 
    - Here we could have used more advanced techniques. Linear regression with more variables. There's something called signature mehtods which have lately found their application in finance. They capture essential geometric and topological properties in the data.

We now have a time series of skew for each ticker. For each ticker we
- Regress skew of individual stocks on the sector skew
- We calculate the residuals of the regression and compute rolling z scores
- We enter when the z score is above a certain threshold and exit when it comes back down to the exit threshold
- Specifically if the spread is high we long stock skew reversal and short sector reversal weighted by the regression coefficient
- We then do delta hedging of both underlyings

## Results
- plots from the results