import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F


def plot_single_day_profile(df, target_date, trim_outliers=True):
    """
    Plots a multi-panel intraday profile (OFI, Spread, Volatility) for a specific date.
    If trim_outliers=True, clips y-axes to remove extreme open/close spikes.
    """
    day_df = df.loc[target_date].copy()

    if day_df.empty:
        print(f"No data available for {target_date}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    title_suffix = " (Outliers Trimmed)" if trim_outliers else ""
    fig.suptitle(f"LOB Microstructure Profile - {target_date}{title_suffix}",
                 fontsize=16, fontweight='bold')

    # Panel 1: Raw OFI + 60-second moving average
    ax1.plot(day_df.index, day_df['ofi'], color='lightgray', alpha=0.5, label='Raw 1s OFI')
    ax1.plot(day_df.index, day_df['ofi'].rolling(60).mean(),
             color='blue', linewidth=2, label='60s Moving Avg OFI')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    if trim_outliers:
        ax1.set_ylim(day_df['ofi'].quantile(0.01), day_df['ofi'].quantile(0.99))
    ax1.set_ylabel("Order Flow Imbalance")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Bid-Ask Spread
    ax2.plot(day_df.index, day_df['spread'], color='orange', linewidth=1.5)
    if trim_outliers:
        ax2.set_ylim(0, day_df['spread'].quantile(0.999))
    ax2.set_ylabel("Bid-Ask Spread")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Rolling Volatility
    ax3.plot(day_df.index, day_df['volatility'], color='purple', linewidth=1.5)
    if trim_outliers:
        ax3.set_ylim(0, day_df['volatility'].quantile(0.99))
    ax3.set_ylabel("60s Volatility")
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel("Time of Day")
    plt.tight_layout()
    plt.show()


def plot_daily_distributions(df, feature='ofi', days_to_plot=5, trim_outliers=False):
    """
    Plots violin distributions of a feature across the first N trading days.
    If trim_outliers=True, clips y-axis to the 2nd-98th percentile range.
    """
    df_plot = df.copy()
    df_plot['date_only'] = df_plot.index.date

    unique_days = df_plot['date_only'].unique()[:days_to_plot]
    df_filtered = df_plot[df_plot['date_only'].isin(unique_days)].copy()

    plt.figure(figsize=(14, 6))
    sns.violinplot(data=df_filtered, x='date_only', y=feature, palette='Blues', inner='quartile')

    if trim_outliers:
        lower = df_filtered[feature].quantile(0.02)
        upper = df_filtered[feature].quantile(0.98)
        plt.ylim(lower, upper)

    if feature == 'ofi':
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)

    zoom_tag = " (Zoomed)" if trim_outliers else ""
    plt.title(f"Daily Distribution of {feature.upper()}{zoom_tag} (First {days_to_plot} Days)",
              fontsize=14)
    plt.xlabel("Date")
    plt.ylabel(feature.replace('_', ' ').title())
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fold_comparison(tkan_f1_scores, lstm_f1_scores, save_path=None):
    """
    Bar chart comparing per-fold Macro-F1 scores between TKAN and LSTM.
    """
    folds = np.arange(1, len(tkan_f1_scores) + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(folds - width / 2, tkan_f1_scores, width, label='TKAN', color='#1f77b4')
    rects2 = ax.bar(folds + width / 2, lstm_f1_scores, width, label='LSTM', color='#ff7f0e')

    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_xlabel('Walk-Forward Fold (Day)', fontsize=12)
    ax.set_title('Stability Comparison: Out-of-Sample F1 Score Across Days',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_fold_stability(tkan_scores, lstm_scores, metric_name="Macro-F1"):
    """
    Line chart showing per-fold performance with mean reference lines.
    """
    folds = np.arange(1, len(tkan_scores) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(folds, tkan_scores, marker='o', markersize=8, color='blue', linewidth=2, label='TKAN')
    plt.plot(folds, lstm_scores, marker='s', markersize=8, color='orange', linewidth=2, label='LSTM')

    # Dashed mean-reference lines
    plt.axhline(np.mean(tkan_scores), color='blue', linestyle=':', alpha=0.5)
    plt.axhline(np.mean(lstm_scores), color='orange', linestyle=':', alpha=0.5)

    plt.title(f'Stability Analysis: {metric_name} Score per Fold', fontsize=16, fontweight='bold')
    plt.xlabel('Fold (Forward Day)', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(folds)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_aggregate_metrics(tkan_f1, lstm_f1, tkan_acc, lstm_acc, save_path=None):
    """
    Grouped bar chart comparing mean Macro-F1 and Accuracy with std-dev error bars.
    """
    labels = ['Macro-F1', 'Accuracy']
    x = np.arange(len(labels))
    width = 0.35

    tkan_means = [np.mean(tkan_f1), np.mean(tkan_acc)]
    tkan_stds = [np.std(tkan_f1), np.std(tkan_acc)]
    lstm_means = [np.mean(lstm_f1), np.mean(lstm_acc)]
    lstm_stds = [np.std(lstm_f1), np.std(lstm_acc)]

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, tkan_means, width, yerr=tkan_stds,
                    label='TKAN', color='blue', capsize=5, alpha=0.8)
    rects2 = ax.bar(x + width / 2, lstm_means, width, yerr=lstm_stds,
                    label='LSTM', color='orange', capsize=5, alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Aggregate Performance: TKAN vs. LSTM', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='lower right')

    # Label each bar with its numeric value
    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_learning_curves(tkan_history, lstm_history):
    """
    Side-by-side loss curves (train vs. validation) for TKAN and LSTM.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle('Learning Dynamics: TKAN vs LSTM (Single Fold)', fontsize=16, fontweight='bold')

    # TKAN
    axes[0].plot(tkan_history['train'], label='Train Loss', color='blue', linestyle='--')
    axes[0].plot(tkan_history['val'], label='Validation Loss', color='blue', linewidth=2)
    axes[0].set_title('TKAN Convergence')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Cross Entropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LSTM
    axes[1].plot(lstm_history['train'], label='Train Loss', color='orange', linestyle='--')
    axes[1].plot(lstm_history['val'], label='Validation Loss', color='orange', linewidth=2)
    axes[1].set_title('LSTM Convergence')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_bspline_kan(kan_layer, input_names=None, output_names=None,
                          x_range=(-3.0, 3.0), num_points=100):
    """
    Extracts and plots the learned B-spline activation functions from a KANLinear layer.
    """
    kan_layer.eval()
    device = kan_layer.spline_weight.device

    in_features = kan_layer.in_features
    out_features = kan_layer.out_features

    # Generate evenly spaced input values
    x = torch.linspace(x_range[0], x_range[1], num_points, device=device)
    X_batch = x.unsqueeze(1).expand(-1, in_features)

    with torch.no_grad():
        x_clamped = torch.tanh(X_batch)
        bases = kan_layer.b_splines(x_clamped)
        base_acts = F.silu(X_batch)

    fig, axes = plt.subplots(out_features, in_features,
                             figsize=(in_features * 2.5, out_features * 2.5))
    fig.suptitle("Learned B-Spline KAN Functions (Opening the Black Box)",
                 fontsize=16, y=1.02)

    # Ensure axes is always 2-D for uniform indexing
    if out_features == 1 and in_features == 1:
        axes = np.array([[axes]])
    elif out_features == 1:
        axes = axes[np.newaxis, :]
    elif in_features == 1:
        axes = axes[:, np.newaxis]

    x_np = x.cpu().numpy()

    for o in range(out_features):
        for i in range(in_features):
            ax = axes[o, i]

            # Spline path: weighted sum of B-spline bases
            spline_contrib = torch.matmul(bases[:, i, :], kan_layer.spline_weight[o, i, :])
            # Base path: SiLU activation scaled by base weight
            base_contrib = base_acts[:, i] * kan_layer.base_weight[o, i]

            y_total = (spline_contrib + base_contrib).detach().cpu().numpy()

            ax.plot(x_np, y_total, color='blue', linewidth=2)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.2)

            if o == 0:
                ax.set_title(input_names[i] if input_names else f"In {i}", fontsize=10)
            if i == 0:
                ax.set_ylabel(output_names[o] if output_names else f"Out {o}", fontsize=10)

    plt.tight_layout()
    plt.show()
