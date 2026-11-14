
"""Track 08 Part 2: Effective trials and Sharpe spread analysis."""

import numpy as np
import pandas as pd

from strategies.angelo_closeloc import build_weights
from src.backtest import run_backtest
from src.contrib.christine import effective_trials, sharpe_spread, bootstrap_hourly_sharpes
from src.stats import deflated_sharpe_ratio


# Research settings
SPANS = (1, 6, 24)
DEADBANDS = (0.005, 0.015, 0.03)
PRIMARY_PARAMS = (6, 0.015)

PERIODS_PER_YEAR = 8760
DEFAULT_TRIALS = 10
DEFAULT_VARIANCE = 0.5 / PERIODS_PER_YEAR

BLOCK_SIZE = 24
N_BOOTSTRAP = 2000
SEED = 42


def run_grid(df, fee_bps=0.0):
    """Run all nine strategy parameter combinations."""
    grid_results = {}

    for span in SPANS:
        for deadband in DEADBANDS:

            weights = build_weights(
                df=df,
                span=span,
                deadband=deadband
            )

            result = run_backtest(
                df=df,
                weights=weights,
                fee_bps=fee_bps
            )

            grid_results[(span, deadband)] = result

    return grid_results


def hourly_sharpe(returns):
    """Calculate the unannualized sample Sharpe ratio."""
    return np.mean(returns) / np.std(returns, ddof=1)



def print_grid_results(grid_results):
    """Display the strategy grid results."""

    print("\nSTRATEGY GRID RESULTS")
    print("-" * 60)

    print(
        f"{'Span':>6} "
        f"{'Deadband':>10} "
        f"{'Sharpe':>10} "
        f"{'DSR':>10} "
        f"{'Turnover':>10}"
    )

    for (span, deadband), result in grid_results.items():

        print(
            f"{span:>6} "
            f"{deadband:>10.3f} "
            f"{result.sharpe:>10.3f} "
            f"{result.deflated_sharpe:>10.4f} "
            f"{result.mean_turnover:>10.4f}"
        )


def main():

    # 1. Load training data
    df = pd.read_parquet("data/train_1h.parquet")

    df = df.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    print("TRAINING DATA")
    print("Dataset shape:", df.shape)
    print("Number of symbols:", df["symbol"].nunique())
    print("Number of hours:", df["timestamp"].nunique())

    # 2. Run the zero-fee strategy grid
    grid_results = run_grid(df, fee_bps=0.0)

    print_grid_results(grid_results)

    # Exclude constant-return strategies from statistical analysis
    valid_results = {}

    for params, result in grid_results.items():

        r = result.hourly_returns

        if not np.isfinite(r).all():
            raise ValueError(
                f"Non-finite returns for {params}"
            )

        if np.all(r == r[0]):
            print(f"Excluding constant-return strategy: {params}")
            continue

        valid_results[params] = result

    print("\nStrategies attempted:", len(grid_results))
    print("Strategies included:", len(valid_results))

    # 3. Construct the strategy returns matrix
    hourly_returns = np.column_stack([
        result.hourly_returns
        for result in valid_results.values()
    ])

    print("Returns matrix shape:", hourly_returns.shape)

    # 4. Estimate effective trials and Sharpe spread
    n_eff = effective_trials(hourly_returns)
    measured_variance = sharpe_spread(hourly_returns)

    rounded_trials = max(1, round(n_eff))

    print("\nEMPIRICAL ESTIMATES")
    print("-" * 60)

    print(f"Effective trials: {n_eff:.6f}")
    print(f"Rounded effective trials: {rounded_trials}")
    print(f"Measured hourly variance: {measured_variance:.8e}")
    print(f"Default hourly variance: {DEFAULT_VARIANCE:.8e}")

    # 5. Evaluate the pre-selected primary strategy
    primary = valid_results[PRIMARY_PARAMS]

    primary_dsr_default = deflated_sharpe_ratio(
        returns=primary.hourly_returns
    )

    primary_dsr_single = deflated_sharpe_ratio(
        returns=primary.hourly_returns,
        n_trials=1
    )

    print("\nPRIMARY STRATEGY")
    print("-" * 60)

    print("Parameters:", PRIMARY_PARAMS)
    print(f"Annualized Sharpe: {primary.sharpe:.3f}")
    print(f"Default DSR: {primary_dsr_default:.4f}")
    print(f"Single-trial DSR: {primary_dsr_single:.4f}")

    # 6. Identify the best strategy using unrounded Sharpe
    sharpe_ratios = {
        params: hourly_sharpe(result.hourly_returns)
        for params, result in valid_results.items()
    }

    best_params = max(
        sharpe_ratios,
        key=sharpe_ratios.get
    )

    best = valid_results[best_params]

    best_dsr_default = deflated_sharpe_ratio(
        returns=best.hourly_returns
    )

    best_dsr_trials = deflated_sharpe_ratio(
        returns=best.hourly_returns,
        n_trials=rounded_trials
    )

    best_dsr_both = deflated_sharpe_ratio(
        returns=best.hourly_returns,
        n_trials=rounded_trials,
        sr_variance_across_trials=measured_variance
    )

    print("\nBEST STRATEGY")
    print("-" * 60)

    print("Parameters:", best_params)
    print(f"Annualized Sharpe: {best.sharpe:.3f}")
    print(f"Default DSR: {best_dsr_default:.4f}")
    print(f"Measured trials DSR: {best_dsr_trials:.4f}")
    print(f"Both measured inputs DSR: {best_dsr_both:.4f}")

    # 7. Estimate null Sharpe variance using block bootstrap
    bootstrap_sharpes = bootstrap_hourly_sharpes(
        primary.hourly_returns
    )

    bootstrap_variance = np.var(
        bootstrap_sharpes,
        ddof=1
    )

    print("\nBLOCK BOOTSTRAP")
    print("-" * 60)

    print(f"Block length: {BLOCK_SIZE}")
    print(f"Number of samples: {N_BOOTSTRAP}")
    print(f"Random seed: {SEED}")

    print(
        f"Mean bootstrap Sharpe: "
        f"{np.mean(bootstrap_sharpes):.8f}"
    )

    print(
        f"Bootstrap Sharpe variance: "
        f"{bootstrap_variance:.8e}"
    )

    print("\nSHARPE VARIANCE COMPARISON")
    print("-" * 60)

    print(f"Default: {DEFAULT_VARIANCE:.8e}")
    print(f"Cross-strategy: {measured_variance:.8e}")
    print(f"Bootstrap null: {bootstrap_variance:.8e}")

    print(
        f"Bootstrap / Default: "
        f"{bootstrap_variance / DEFAULT_VARIANCE:.4f}"
    )

    print(
        f"Measured / Bootstrap: "
        f"{measured_variance / bootstrap_variance:.4f}"
    )

    # 8. Verify transaction fee sensitivity
    fee_results = run_grid(df, fee_bps=5.0)

    print("\nTRANSACTION FEE CHECK (5 BPS)")
    print("-" * 60)

    print_grid_results(fee_results)

    all_negative = all(
        hourly_sharpe(result.hourly_returns) < 0
        for result in fee_results.values()
    )

    all_zero_dsr = all(
        result.deflated_sharpe == 0.0
        for result in fee_results.values()
    )

    print("\nAll strategies have negative Sharpe:", all_negative)
    print("All default DSR values are zero:", all_zero_dsr)


if __name__ == "__main__":
    main()
