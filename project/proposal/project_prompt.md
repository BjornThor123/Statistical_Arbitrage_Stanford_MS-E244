# Statistical arbitrage project

I am doing a group project in a graduate course called statistical arbitrage. The goal is to analyze data and build a trading strategy. We are not evaluated by how much money the strategy makes or by the sharpe. We are evaluated on creativity and how we are using concepts from the class. Some of the things we have covered includes options, volatility, equitities, convertible bonds, pairs trading, factor models, carry trade, crypto, and more. We are very free in choosing the project. We have about 1 month to finish the project. Right now we have two main ideas:

- Options trading for mispricing in volatility
- Hedging against skew in pairs trading

Here is a description of what we envision for each of the projects:

## Options trading for mispricing in volatility
In broad terms we want to model realized volatility and look at relative mispricings at the cross section of options. We may want to do this close to ATM since that is were the trading volume is. This requires us to model realized volatility which can be done in several ways. We can for instance use different versions of GARCH-models, i.e. GJR-GARCH to capture tail-behavior. We can also use more complex models like xgboost or LSTMs. We can also use the output of a GARCH as input to an xgboost model to improve on it so that our model can consider other metadata about the market. We would then look at where the realized and implied volatility differs the most and trade based on this.

## Hedging against skew in pairs trading
This is a very interesting idea which I haven't seen in the literature. I am not certain that everything I say to explain this idea is true, so you would need to check that my assumptions for this project holds. First, we note that pairs trading is trading based on some cointegrated series, which is quite close to trading covariance. This strategy can achieve a good sharpe, however, from what I know it can suffer from big drawdowns when the covariance structure of the market breaks down. This shows up as skew in the strategies' return distribution. One could hedge this by buying puts and calls to hedge against big spikes. However, this might be expensive. So instead of only calculating pairs to trade covariance, we can also calculate the skew between a triple of assets. The idea then is to do pairs trading plus trade in the third asset to offset the skew and then be skew-neutral. As mentioned, I am not very sure about these assumptions, so I need you to check that this project can make sense at all. 


## Requirements for the proposal
I want you to draft a project proposal based on the ideas above. You can choose to write a proposal for one of the ideas above, or you can choose to combine them. Combining them would be very interesting, but don't do it if you think it is a bad idea. We don't want it to be forced. We would prefer to do a project involving volatility and options.

The project proposal should satisfy the following requirements:
- The introduction should explain the project in broad terms.
- It should list imoportant and relevant papers we can draw inspiration from.
- It should suggest a few different methods we can use (for instance xgboost for modeling realized volatility).
- If applicable it should contain mathematical models.
- List the data we would need to successfully complete the project.
- The project proposal should split the project into three phases
    - First phase - proof of concept: this should be fairly easy to implement and backtest. A relatively simple model as a 
    - Second phase - more complex scenario: add complexities and frictions similar to real world trading. For instance transaction cost, not being able to trade certain instruments, delay in trading such that we don't get the price we are making the decision based on. We should still be able to complete this phase. So, some of the frictions mentioned above might be a bit unrealistic to be able to address in our project given the timeframe.
    - Third phase - realistic production: add more complexities of real world trading and real world data. Here we should describe what we would like to do if we had 6 months or more to do the project. What more could we explore? Here you can include any frictions you would like. For instance adding constraints on how much we can trade as would be the case for a portfolio manager. Some investors might want to withdraw money so we would need to consider how this would affect the strategy. 
- You can assume we have expert knowledge in quantitative finance
- Output in markdown format
