import pandas as pd
from src.config import get_config
from src.data_loader import DataLoader
from src.backtest1 import run_backtest
from src.run_strategy import run_strategy

config = get_config()


# ── Signal construction ──────────────────────────────────────────────────────

def compute_signals(
    resid_df: pd.DataFrame,
    z_threshold: float = 1.0,
    signal_window: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalise idiosyncratic residuals into rolling z-scores, then threshold:

        z_{i,t} = (ε_{i,t} - μ_ε) / σ_ε   (rolling over signal_window days)

    Signal conventions
    ------------------
    +1  sell skew : z > +z_threshold → stock skew unusually steep vs sector
                    → puts overpriced → sell put spread / risk reversal
    -1  buy skew  : z < -z_threshold → stock skew unusually flat vs sector
                    → puts cheap → buy put spread / risk reversal
     0  flat

    Returns
    -------
    (signals, z_scores) – both DataFrames indexed like resid_df.
    """
    rolling_mean = resid_df.rolling(signal_window).mean()
    rolling_std  = resid_df.rolling(signal_window).std()
    z_scores     = (resid_df - rolling_mean) / rolling_std

    signals = pd.DataFrame(0, index=resid_df.index, columns=resid_df.columns)
    signals[z_scores >  z_threshold] =  1   # sell skew
    signals[z_scores < -z_threshold] = -1   # buy skew

    return signals, z_scores


def main():
    loader = DataLoader(data_path=config.data_path)

    query = (
        f"SELECT {', '.join(config.relevant_option_columns)} FROM options_enriched"
        f" WHERE date >= '{config.start_date}' AND date <= '{config.end_date}'"
        f" AND tte_days <= {config.max_tte}"
    )
    df = loader.query(query)

    results = run_strategy(df, compute_signals=compute_signals)

    print("\nSignal counts per ticker:")
    print(
        results["signals"]
        .apply(lambda col: col.value_counts())
        .T.fillna(0).astype(int)
    )

    backtest_results = run_backtest(
        signals=results["signals"],
        spot_prices=results["spot_prices"],
        z_scores=results["z_scores"],
        plot_dir="project/plots",
    )

    return {**results, **backtest_results}


if __name__ == '__main__':
    main()
