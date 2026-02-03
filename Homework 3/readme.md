# Comments on HW3

## Question 1
- Part 1: I don't understand the calculation here. Is $R_t^A$ supposed to be $P_t^A$?
- Part 2: Should we say something about Moore-Penrose pseudo-inverse?
- Part 2: Should we use gradient notation instead?

## Question 2
In the code $W$ is defined as `W = b.T @ np.linalg.inv(b @ b.T)`. I think it should be:
```python
W = np.array([self.metadata.loc[returns.columns, 'Sector'] == sector for sector in self.sectors]).T
W = W/np.sum(W, axis=0)
```
That is the only difference between our solutions.
The thing is that we get slightly different `comp_mtx_df`. I think it is due to numerical instability in taking the inverse.

## Question 4
For some reason we have slightly different residuals.

## Question 5
The histograms looks different.
Why do we 'check polarity' and flip signs in `raw_weights_f1`.

