import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chisquare, mstats


def get_dynamic_colors(df, group_column):
    """
    Generates a color palette for unique values in a group column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_column (str): The column name to group by.

    Returns:
        dict: A dictionary mapping each unique group value to a color.
    """
    unique_groups = df[group_column].unique()
    cmap = plt.get_cmap('coolwarm')
    colors = {
        group: cmap(i / (len(unique_groups) - 1))
        for i, group in enumerate(unique_groups)
    }
    return colors


def del_outliers(df, group_column='Promotion', value_column='SalesInThousands'):
    """
    Replaces outliers in a DataFrame grouped by a specified column
    with the corresponding upper or lower limit.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_column (str): The column name to group the data by.
        value_column (str): The column name from which outliers should be
        replaced.

    Returns:
        pd.DataFrame: The DataFrame with outliers replaced by thresholds.
    """

    result_dfs = []
    for group_value in df[group_column].unique():
        group_df = df[df[group_column] == group_value].copy()

        q1 = group_df[value_column].quantile(0.25)
        q3 = group_df[value_column].quantile(0.75)
        iqr = q3 - q1
        low_limit = q1 - 1.5 * iqr
        up_limit = q3 + 1.5 * iqr

        group_df[value_column] = group_df[value_column].clip(
            lower=low_limit, upper=up_limit
        )

        result_dfs.append(group_df)

    return pd.concat(result_dfs, ignore_index=True)


def heatmap_mean_values(
    df,
    group_column1='Promotion',
    group_column2='MarketSize',
    value_column='SalesInThousands',
    cmap='coolwarm',
):
    """
    Creates a heatmap visualizing the mean of a value column by two categorical
    columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_column1 (str): The column name for the rows of the heatmap.
        group_column2 (str): The column name for the columns of the heatmap.
        value_column (str): The column name for calculating the mean.
        cmap (str, optional): The name of the Matplotlib colormap to use.
        Defaults to 'coolwarm'.
    """
    mean_data = (
        df.groupby([group_column1, group_column2])[value_column].mean().reset_index()
    )

    heatmap_data = mean_data.pivot(
        index=group_column1, columns=group_column2, values=value_column
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap=cmap)
    plt.title(
        f'Mean {value_column} by {group_column1} and {group_column2}',
        fontsize=14
    )
    plt.xlabel(group_column2, fontsize=12)
    plt.ylabel(group_column1, fontsize=12)
    plt.show()


def qq_plot_all(data, group_col='Promotion', value_col='SalesInThousands',
                colors=None):
    """
    Creates QQ plots with confidence bands and a legend for each group
    in a DataFrame, displayed as subplots.

    Args:
        data: DataFrame containing the data.
        group_col: Name of the column specifying the groups.
        value_col: Name of the column containing the values.
        colors (optional): List of colors for each group's points
        (default: None).
    """
    groups = data[group_col].unique()
    if colors is None:
        colors = get_dynamic_colors(data, group_col)

    fig, axes = plt.subplots(nrows=1, ncols=len(groups), figsize=(15, 6))
    fig.subplots_adjust(wspace=0.4)

    for i, group in enumerate(groups):
        group_data = data[data[group_col] == group][value_col]
        sm.qqplot(
            group_data,
            line='s',
            fit=True,
            ax=axes[i],
            marker='o',
            markerfacecolor=colors[group],
            markeredgecolor='black',
            markersize=8,
            alpha=0.7,
            label=f'Promotion {group}',
        )

        axes[i].set_title(f'Promotion: {group}', fontsize=14)
        axes[i].set_xlabel('Theoretical Quantiles', fontsize=12)
        axes[i].set_ylabel('Sample Quantiles', fontsize=12)
        axes[i].grid(alpha=0.4)

    fig.legend(title='Promotion', fontsize=10)

    plt.show()


def shapiro_test(data, group_column='Promotion',
                 value_column='SalesInThousands', alpha=0.05):
    """
    Performs the Shapiro-Wilk normality test for each group specified in a
    DataFrame.
    The test assesses whether the distribution of values within each group is
    likely to be normally distributed.

    Args:
        data: pandas DataFrame containing the data to be analyzed.
        group_column: str (default: 'Promotion'). The name of the column in the
        DataFrame that defines the groups for the test.
        value_column: str (default: 'SalesInThousands'). The name of the column
        containing the numeric values to be tested for normality
        within each group.
        alpha: float (default: 0.05). The significance level for the test.
        This determines the threshold p-value for rejecting the null hypothesis
        of normality.

    Returns:
        None. The function prints the results of the Shapiro-Wilk test for
        each group, including the test statistic (W), p-value, and a conclusion
        about whether the distribution is likely normal or not.
    """

    groups = data[group_column].unique()

    print(
        '\nShapiro-Wilk Test Results:\n'
        '-------------------------------------------------------'
    )
    for group in groups:
        group_data = data[data[group_column] == group][value_column]
        statistic, p_value = stats.shapiro(group_data)
        print(f'Promotion {group}: W = {statistic:.3f},' f' p-value = '
              f'{p_value:.10f}')

        if p_value > alpha:
            print(f' H0 hypothesis accepted:' f' Promotion {group} '
                  f'is likely normal.')
        else:
            print(
                f' H0 hypothesis rejected: '
                f'Promotion {group} is likely not normal.\n'
            )


def kruskal_test(data, group_col='Promotion', value_col='SalesInThousands',
                 alpha=0.05):
    """
    Performs the Kruskal-Wallis H-test for independent samples.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data to be
        analyzed.
        group_col (str, optional): The name of the column in the DataFrame
        that defines the groups. Defaults to 'Promotion'.
        value_col (str, optional): The name of the column containing the numeric
        values to be tested. Defaults to 'SalesInThousands'.
        alpha (float, optional): The significance level for the test. Default
        to 0.05.

    Returns:
        None: The function prints the results of the Kruskal-Wallis test,
        including the test statistic (H), p-value, and a conclusion about
        whether there are significant differences between the groups.
    """

    groups = data[group_col].unique()
    group_data = [data[data[group_col] == group][value_col].values for group in groups]

    print('\nKruskal-Wallis Test Results:')
    print('-------------------------------------------------------')
    statistic, p_value = mstats.kruskalwallis(*group_data)
    print(f'H-statistic: {statistic:.4f}, p-value: {p_value:.4f}')

    if p_value > alpha:
        print('H0 hypothesis accepted: No significant differences between groups.')
    else:
        print('H0 hypothesis rejected: Significant differences between groups.')


def pairwise_test(df, group_column='Promotion', value_column='SalesInThousands'):
    """
    Performs pairwise Mann-Whitney U tests with Bonferroni correction for
    multiple comparisons.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_column (str, optional): The column name that defines the groups.
        Defaults to 'Promotion'.
        value_column (str, optional): The column name containing the values for
        comparison. Defaults to 'SalesInThousands'.
    """

    groups = df[group_column].unique()
    num_comparisons = (
        len(groups) * (len(groups) - 1) // 2
    )

    print('\nPairwise Mann-Whitney U Test Results '
          '(with Bonferroni Correction):')
    print('-------------------------------------------------------------------')

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = df[df[group_column] == groups[i]][value_column]
            group2 = df[df[group_column] == groups[j]][value_column]

            try:
                u_statistic, p_value = stats.mannwhitneyu(group1, group2)
            except ValueError as e:
                print(
                    f'Error for {groups[i]} vs. {groups[j]}:'
                    f' {e} (likely due to insufficient data)'
                )
                continue

            p_value_adjusted = p_value * num_comparisons
            print(
                f'{group_column} {groups[i]} vs. {group_column} {groups[j]}:'
                f' U = {u_statistic:.4f}, p-value (adjusted) ='
                f' {p_value_adjusted:.4f}'
            )


def check_df(dataframe, head=5, transpose=True):
    """
    Prints a summary of key information about a Pandas DataFrame.

    Args:
        dataframe: The DataFrame to analyze.
        head: Number of rows to display from the beginning and
        end of the DataFrame (default: 5).
        transpose: Whether to transpose the quantile output (default: True).
    """

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.width', 500)
    print('############## Shape ##############')
    print(dataframe.shape)
    print('############## Types ##############')
    print(dataframe.dtypes)
    print('############## Head ##############')
    print(dataframe.head(head))
    print('############## Tail ##############')
    print(dataframe.tail(head))
    print('############## NA ##############')
    print(dataframe.isnull().sum())
    print('############## Quantiles ##############')
    quantiles = dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1])
    if transpose:
        print(quantiles.T)
    else:
        print(quantiles)


def histogram_group(
    data,
    group_column='Promotion',
    value_column='SalesInThousands',
    colors=None,
    bins=20,
    kde=True,
    log_transform=False,
):
    """
    Creates histograms with optional KDE and log transformation for groups
    within a DataFrame.

    Args:
        data: The pandas DataFrame containing the data.
        group_column: The column name used to define groups.
        value_column: The column name containing the values for analysis.
        colors: List of colors for each group
        (default: None, uses a default palette).
        bins: Number of bins for the histograms (default: 10).
        kde: Whether to add kernel density estimation plots (default: True).
        log_transform: Whether to log-transform the values before plotting
        (default: False).
    """

    groups = data[group_column].unique()
    if colors is None:
        colors = get_dynamic_colors(data, group_column)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, group in enumerate(groups):
        group_data = data[data[group_column] == group][value_column]

        if log_transform:
            group_data = np.log1p(group_data)

        sns.histplot(
            group_data, bins=bins, kde=kde, color=colors[group], label=group,
            ax=ax
        )

    ax.set_title('Histogram data by Group', fontsize=14)
    ax.set_xlabel(
        f'{value_column} (log-transformed)' if log_transform else value_column,
        fontsize=12,
    )
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(title=f'{group_column}:', fontsize=10)
    ax.grid(alpha=0.4)
    plt.show()


def boxplot_group(
    df,
    group_column='Promotion',
    value_column='SalesInThousands',
    colors=None,
    title=None,
):
    """
    Creates a Seaborn boxplot with a touch of pizzazz.

    Args:
        df: The DataFrame containing the data.
        group_column: The column name for grouping (e.g., 'Promotion').
        value_column: The column name for the values to plot
        (e.g., 'SalesInThousands').
        colors: A list of colors for each group
        (optional, defaults to a vibrant palette).
        title: The title of the plot
        (optional, defaults to 'Boxplot Bonanza by Group').
    """

    if colors is None:
        colors = get_dynamic_colors(df, group_column)

    plt.figure(figsize=(10, 8))

    sns.boxplot(
        x=group_column,
        y=value_column,
        hue=group_column,
        data=df,
        palette=colors,
        legend=False,
    )

    plt.title(title)
    plt.xlabel(group_column, fontsize=12)
    plt.ylabel(value_column, fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()


def analyze_categorical_variable(
    df, group_column='Promotion', value_column='SalesInThousands', colors=None
):
    """
    Calculates group ratios, performs a chi-squared test for SRM, and
    visualizes the results with a count plot and a table including counts.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_column (str): The column name to group by.
        value_column (str): The column name to count and calculate
        unique values from.
        colors (list): A list of colors for each group (optional).
    """

    grouped_data = df.groupby(group_column)[value_column].agg(
        Count='count', Unique_Values='nunique'
    )

    total_count = grouped_data['Count'].sum()
    grouped_data['Ratio'] = grouped_data['Count'] / total_count
    grouped_data['Total_Count'] = total_count

    if value_column == group_column:
        print(f'\nStatistics of \'{value_column}\':\n')
    else:
        print(f'\nGroup column \'{group_column}\' statistics:\n')
    print(grouped_data.to_markdown(numalign='left', stralign='left',
                                   floatfmt='.4f'))

    plt.figure(figsize=(8, 5))
    if colors is None:
        colors = get_dynamic_colors(df, group_column)

    sns.countplot(
        x=df[group_column], data=df, palette=colors, hue=group_column,
        legend=False
    )

    plt.title(f'Counts of \'{group_column}\' values', fontsize=14)
    plt.xlabel(group_column, fontsize=12)
    plt.ylabel('Count', fontsize=12)

    plt.show()


def chi_squared_test_with_sampling(
    df, group_column, count_column, sample_size=1500, alpha=0.05
):
    """
    Performs a chi-squared test for equal distribution after sampling from a
    DataFrame.

    Args:
        df: The DataFrame containing the data.
        group_column: The column name used for grouping categories.
        count_column: The column name containing the counts for each category.
        sample_size: The desired sample size (default: 1000).
        alpha: The desired significance level.
    Returns:
        chi2_stat: The chi-squared test statistic.
        p_value: The p-value of the test.
    """

    sampled_df = df.sample(n=sample_size, replace=True)

    grouped_data = (
        sampled_df.groupby(group_column)[count_column].sum().reset_index())

    observed_counts = grouped_data[count_column]
    total_count = observed_counts.sum()
    expected_counts = [total_count / len(grouped_data)] * len(grouped_data)

    chi2_stat, p_value = chisquare(observed_counts, expected_counts)

    if p_value > alpha:
        print('H0 hypothesis accepted: Equal distribution between groups.')
    else:
        print('H0 hypothesis rejected: Not equal distribution between groups.')
    print(f'Chi-squared Statistic: {chi2_stat:.4f}, P-value: {p_value:.4f}')

    return sampled_df

def bootstrap_retention_rates(df, n_iterations=1500, plot=True,
                              confidence_level=0.95):
    """
    Performs bootstrapping to estimate the distribution of 1-day and 7-day
    retention rates.

    Args:
        df: The DataFrame containing the data (must have 'version',
        'retention_1', and 'retention_7' columns).
        n_iterations: The number of bootstrap iterations to perform.
        plot: Whether to plot the KDEs of the bootstrap distributions.
        confidence_level: The confidence level of the bootstrap distributions.

    Returns:
        boot_1d_df: DataFrame containing the bootstrap distribution of 1-day
        retention rates.
        boot_7d_df: DataFrame containing the bootstrap distribution of 7-day
        retention rates.
    """
    boot_1d = []
    boot_7d = []
    for _ in range(n_iterations):
        boot_sample = df.sample(frac=1, replace=True)
        boot_mean_1 = boot_sample.groupby('version')['retention_1'].mean()
        boot_mean_7 = boot_sample.groupby('version')['retention_7'].mean()
        boot_1d.append(boot_mean_1)
        boot_7d.append(boot_mean_7)

    boot_1d_df = pd.DataFrame(boot_1d)
    boot_7d_df = pd.DataFrame(boot_7d)

    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_1d = {
        version: np.percentile(boot_1d_df[version], [lower_percentile,
                                                     upper_percentile])
        for version in df['version'].unique()
    }

    ci_7d = {
        version: np.percentile(boot_7d_df[version], [lower_percentile,
                                                     upper_percentile])
        for version in df['version'].unique()
    }

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                       figsize=(13, 5))

        boot_1d_df.plot.kde(ax=ax1)
        ax1.set_xlabel('retention rate', size=10)
        ax1.set_ylabel('density', size=12)
        ax1.set_title('1-day retention rate distribution', fontweight='bold',
                      size=14)

        boot_7d_df.plot.kde(ax=ax2)
        ax2.set_xlabel('retention rate', size=10)
        ax2.set_title('7-day retention rate distribution', fontweight='bold',
                      size=14)
        plt.show()

    return ci_1d, ci_7d


def analytical_retention_cis(df, confidence_level=0.95):
    """
    Calculates mean retention rates, sample sizes, and analytic confidence
    intervals for
    1-day and 7-day retention rates. Optionally plots histograms of the
    retention rates.

    Args:
        df: The DataFrame containing the data ('version', 'retention_1',
        'retention_7').
        confidence_level: The desired confidence level (default 95%).

    Returns:
        retention_stats: DataFrame containing mean retention rates and sample
        sizes per version.
        ci_1d: A dictionary containing the lower and upper confidence intervals
        for 1-day retention.
        ci_7d: A dictionary containing the lower and upper confidence intervals
        for 7-day retention.
    """


    def calculate_analytic_ci(p, n, confidence_level):
        z = st.norm.ppf(1 - (1 - confidence_level) / 2)
        se = np.sqrt(p * (1 - p) / n)
        return p - z * se, p + z * se


    retention_stats = df.groupby('version')[['retention_1',
                                             'retention_7']].agg(['mean',
                                                                  'count'])
    mean_1d = retention_stats['retention_1']['mean']
    mean_7d = retention_stats['retention_7']['mean']
    n_samples = retention_stats['retention_1']['count']

    ci_1a = {
        version: calculate_analytic_ci(mean_1d[version], n_samples[version],
                                       confidence_level)
        for version in mean_1d.index
    }
    ci_7a = {
        version: calculate_analytic_ci(mean_7d[version], n_samples[version],
                                       confidence_level)
        for version in mean_7d.index
    }

    return ci_1a, ci_7a
