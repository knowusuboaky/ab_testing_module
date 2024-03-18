import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power, zt_ind_solve_power, NormalIndPower
from statsmodels.stats.power import GofChisquarePower
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
import warnings
import plotly.express as px


pd.set_option('display.max_colwidth', None)  # For pandas versions < 1.0.0, use -1 instead of None
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




### Helper Functions ###
def cohen_d(x, y):
    """Calculate Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def phi_coefficient(chi2, n):
    """Calculate Phi coefficient for a 2x2 contingency table."""
    return sqrt(chi2 / n)

def r_effect_size(z_stat, n):
    """Calculate r, the effect size for Z and U statistics."""
    return z_stat / np.sqrt(n)

def cohen_h(p1, p2):
    """Calculate Cohen's h, an effect size used for comparing proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def odds_ratio(contingency_table):
    """Calculate the odds ratio for a 2x2 contingency table."""
    a, b, c, d = contingency_table.flatten()
    return (a * d) / (b * c)

def calculate_eta_squared(anova_table):
    ss_between = anova_table['sum_sq']['C(group)']  # Sum of squares Between groups
    ss_total = ss_between + anova_table['sum_sq']['Residual']  # Total Sum of Squares = SSB + SSE
    eta_squared = ss_between / ss_total  # Eta Squared Calculation
    return eta_squared

### Other ###
def power_analysis_chi_square(effect_size, n, alpha=0.05):
    """Estimate power for a Chi-square test."""
    power_analysis = GofChisquarePower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs=n, alpha=alpha, n_bins=2)
    return power

def power_analysis_ztest(p1, p2, n1, n2, alpha=0.05):
    """Calculate power of a Z-test given two proportions and sample sizes."""
    effect_size = cohen_h(p1, p2)
    power_analysis = NormalIndPower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs1=n1, alpha=alpha, ratio=n2/n1, alternative='two-sided')
    return power

def power_analysis_ttest(d, n1, n2, alpha=0.05):
    """Calculate power of a t-test given Cohen's d, sample sizes, and alpha."""
    power_analysis = TTestIndPower()
    power = power_analysis.solve_power(effect_size=d, nobs1=n1, alpha=alpha, ratio=n2/n1, alternative='two-sided')
    return power

def power_analysis_mann_whitney(u_stat, n1, n2, alpha=0.05):
    """Estimate power for Mann-Whitney U test using normal approximation for large samples."""
    z = u_stat / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    power = stats.norm.cdf(z - stats.norm.ppf(1 - alpha / 2))
    return power


def calculate_power(effect_size, alpha, k_groups, n_per_group):
    """Calculate power of the ANOVA test."""
    n_total = n_per_group * k_groups  # Calculate total sample size
    analysis = FTestAnovaPower()
    power = analysis.solve_power(effect_size=effect_size, nobs=n_total, alpha=alpha, k_groups=k_groups)
    return power

def perform_kruskal_wallis_test_and_effect_size(data, group_column, value_column):
    # Extract unique groups
    groups = data[group_column].unique()
    
    # Prepare data for test - create a list of values for each group
    data_for_test = [data[value_column][data[group_column] == group] for group in groups]
    
    # Perform Kruskal-Wallis test
    stat, p_value = kruskal(*data_for_test)
    
    # Calculate effect size (Eta Squared for Kruskal-Wallis)
    k = len(groups)  # Number of groups
    N = len(data)  # Total number of observations
    eta_squared = (stat - (k - 1)) / (N - k)
    
    return stat, p_value, eta_squared

def simulate_kruskal_wallis_power(distributions, alpha=0.05, num_simulations=1000):
    num_rejections = 0
    for _ in range(num_simulations):
        samples = [np.random.normal(loc=loc, scale=scale, size=n) for n, loc, scale in distributions]
        _, p_value = kruskal(*samples)
        if p_value < alpha:
            num_rejections += 1
    return num_rejections / num_simulations



### Outliers ###
def handle_outliers_method(data, column, method='IQR', strategy='remove'):
    """Outlier detection and handling in a specified column of the DataFrame."""
    if method == 'IQR':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'Z-score':
        mean = data[column].mean()
        std = data[column].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")

    # Identify outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    # Handle outliers
    if strategy == 'remove':
        return data[~data.index.isin(outliers.index)]
    elif strategy == 'impute':
        data.loc[outliers.index, column] = data[column].median()  # Example: impute with median
    elif strategy == 'cap':
        data.loc[data[column] < lower_bound, column] = lower_bound
        data.loc[data[column] > upper_bound, column] = upper_bound
    return data


def validate_input(data, group_column, value_column):
    """Validates the input DataFrame, group column, and value column."""
    assert group_column in data.columns, f"Group column '{group_column}' not found in DataFrame."
    assert value_column in data.columns, f"Value column '{value_column}' not found in DataFrame."
    # Check if the value column is numeric
    assert pd.api.types.is_numeric_dtype(data[value_column]), f"Value column '{value_column}' must be numeric."

### Comparison ###
def perform_posthoc_tukey(data, group_column, value_column, control_group):
    """Perform post-hoc Tukey HSD test to compare all groups against the control."""
    mc = MultiComparison(data[value_column], data[group_column])
    tukey_result = mc.tukeyhsd()
    control_comparisons = tukey_result.summary().data[1:]  # Skip the header row
    control_comparisons = [row for row in control_comparisons if control_group in row[0]]
    return control_comparisons

def perform_mann_whitney_pairwise(data, group_column, value_column, control_group):
    """Perform pairwise Mann-Whitney U tests comparing each group against the control."""
    comparisons = []
    groups = data[group_column].unique()
    control_data = data[data[group_column] == control_group][value_column]
    for group in groups:
        if group != control_group:
            group_data = data[data[group_column] == group][value_column]
            u_stat, p_value = mannwhitneyu(control_data, group_data, alternative='two-sided')
            comparisons.append((group, u_stat, p_value))
    return comparisons

### Interpretation ###
def interpret_test_results(p_value, test_statistic, test_name, effect_size=None, power=None):
    """Interpret the results of a statistical test."""
    significance_level = "this level"
    evidence = "insufficient evidence to reject the null hypothesis"
    if p_value < 0.01:
        significance_level = "1% level"
        evidence = "strong evidence against the null hypothesis"
    elif p_value < 0.05:
        significance_level = "5% level"
        evidence = "evidence against the null hypothesis"
    
    interpretation = (f"For the {test_name}, a test statistic of {test_statistic:.4f} and a p-value of {p_value:.4f} " 
                      f"provides {evidence} at the {significance_level}.")
    if effect_size is not None:
        interpretation += f" The effect size is {effect_size:.4f}."

        # Adding power to the interpretation, if provided
    if power is not None:
        interpretation += f" The power of the test is {power:.4f}, indicating "
        if power < 0.8:
            interpretation += "a potential risk of Type II error (failing to detect a true effect)."
        else:
            interpretation += "a good ability to detect a true effect, if it exists."
    
    return interpretation



### Data Viz ###
def generate_viz_interpretation(data, viz_type, group_column=None, value_column=None):
    """
    Generates a data-driven interpretation for a specific type of visualization.
    
    Parameters:
    - data: DataFrame containing the dataset.
    - viz_type: Type of the visualization ('boxplot', 'violinplot', 'histogram', 'countplot', 'heatmap').
    - group_column: The name of the column used for grouping data (if applicable).
    - value_column: The name of the column representing the values to analyze (if applicable).
    
    Returns:
    - A string containing the interpretation of the data for the specified visualization type.
    """
    interpretation = f"Interpretation of the {viz_type.title()}:\n<br>"
    
    # Boxplot Interpretation
    if viz_type == 'boxplot':
        outliers = data[(data[value_column] > data[value_column].quantile(0.75) + 1.5 * (data[value_column].quantile(0.75) - data[value_column].quantile(0.25))) | 
                        (data[value_column] < data[value_column].quantile(0.25) - 1.5 * (data[value_column].quantile(0.75) - data[value_column].quantile(0.25)))]
        group_outliers = outliers[group_column].value_counts()
        if not group_outliers.empty:
            interpretation += "Outliers indicate variance. "
            interpretation += f"Groups with the most outliers: {', '.join(group_outliers.index)}.\n"
        else:
            interpretation += "No significant outliers, indicating uniformity across groups.\n"
    
    # Violin Plot Interpretation
    elif viz_type == 'violinplot':
        interpretation += "Violin plots reveal distribution shapes. Bimodality or skewness hints at underlying patterns or deviations."
    
    # Histogram Interpretation
    elif viz_type == 'histogram':
        skewness = data[value_column].skew()
        interpretation += f"Skewness ({skewness:.2f}) indicates the distribution's asymmetry. "
        interpretation += "<br>Right skew (>1) shows a tail in higher values; left skew (<-1) in lower values."
    
    # Count Plot Interpretation
    elif viz_type == 'countplot':
        category_counts = data[group_column].value_counts()
        if not category_counts.empty:
            most_common = category_counts.idxmax()
            interpretation += f"Most common category: {most_common}, indicating potential imbalance or predominance."
    
    # Heatmap Interpretation
    elif viz_type == 'heatmap':
        correlations = data.corr()
        strong_corrs = correlations[(correlations.abs() > 0.8) & (correlations.abs() < 1.0)]
        if not strong_corrs.empty:
            interpretation += "Strong correlations (>0.8) suggest significant relationships between variables."
        else:
            interpretation += "No strong correlations detected, indicating variables may independently influence outcomes."
    
    return interpretation


# Assume generate_viz_interpretation function is defined elsewhere in your code

def generate_visualizations(data, group_column, value_column, viz_types):
    """
    Generate specified visualizations with data-driven interpretations.

    Parameters:
    - data: DataFrame containing the dataset.
    - group_column: The name of the column used for grouping data.
    - value_column: The name of the column representing the values to analyze.
    - viz_types: List of visualization types to generate (e.g., ['boxplot', 'heatmap']).
    """
    px.defaults.template = "simple_white"
    
    for viz_type in viz_types:
        if viz_type == 'boxplot':

            # Visualization: Boxplot
            fig_box = px.box(data, x=group_column, y=value_column, color=group_column, title='Data Distribution by Group')
            interpretation = generate_viz_interpretation(data, 'boxplot', group_column, value_column)
            fig_box.add_annotation(xref='paper', yref='paper', x=0.5, y=-0.2,
                                text=f"{interpretation}",
                                showarrow=False, align="center", xanchor='center', yanchor='top')
            fig_box.update_layout(autosize=False, width=1000, height=600, margin=dict(t=50, b=150))
            fig_box.show()
        
        elif viz_type == 'violinplot':
            # Visualization: Violin Plot
            fig_violin = px.violin(data, x=group_column, y=value_column, color=group_column, title='Data Distribution Overview', box=True, points="all")
            interpretation = generate_viz_interpretation(data, 'violinplot', group_column, value_column)
            fig_violin.add_annotation(xref='paper', yref='paper', x=0.5, y=-0.2,
                                    text=f"{interpretation}",
                                    showarrow=False, align="center", xanchor='center', yanchor='top')
            fig_violin.update_layout(autosize=False, width=1000, height=600, margin=dict(t=50, b=150))
            fig_violin.show()
        
        elif viz_type == 'histogram':
            # Visualization: Histogram
            fig_hist = px.histogram(data, x=value_column, color=group_column, marginal="rug", title='Frequency Distribution by Group', barmode='overlay')
            interpretation = generate_viz_interpretation(data, 'histogram', group_column, value_column)
            fig_hist.add_annotation(xref='paper', yref='paper', x=0.5, y=-0.2,
                                    text=f"{interpretation}",
                                    showarrow=False, align="center", xanchor='center', yanchor='top')
            fig_hist.update_layout(autosize=False, width=1000, height=600, margin=dict(t=50, b=150))
            fig_hist.show()

        elif viz_type == 'countplot':
            # Visualization: Count Plot
            fig_count = px.histogram(data, x=group_column, color=group_column, title='Count of Categories')
            interpretation = generate_viz_interpretation(data, 'countplot', group_column, value_column)
            fig_count.add_annotation(xref='paper', yref='paper', x=0.5, y=-0.2,
                                    text=f"{interpretation}",
                                    showarrow=False, align="center", xanchor='center', yanchor='top')
            fig_count.update_layout(autosize=False, width=1000, height=600, margin=dict(t=50, b=150))
            fig_count.show()

        elif viz_type == 'heatmap':
            # Visualization: Heatmap of the Correlation Matrix
            numerical_data = data.select_dtypes(include=[np.number])
            if len(numerical_data.columns) > 1:
                corr_matrix = numerical_data.corr()
                fig_heatmap = px.imshow(corr_matrix, text_auto=True, title="Heatmap of Correlation Matrix", labels=dict(x="Variable", y="Variable", color="Correlation"))
                interpretation = generate_viz_interpretation(data, 'heatmap', group_column, value_column)
                fig_heatmap.add_annotation(xref='paper', yref='paper', x=0.5, y=-0.2,
                                        text=f"{interpretation}",
                                        showarrow=False, align="center", xanchor='center', yanchor='top')
                fig_heatmap.update_layout(autosize=False, width=800, height=800, margin=dict(t=50, b=150))
                fig_heatmap.show()
            else:
                continue  # Skip heatmap if not enough numerical columns
        else:
            continue  # Skip if the viz_type is not recognized



### Modeling ###
def fit_advanced_model(data, formula, model_type='linear'):
    """
    Fits an advanced statistical model based on the provided specifications.
    
    Parameters:
    - data: DataFrame, the dataset for modeling.
    - formula: str, a Patsy formula for the model.
    - model_type: str, specifies the type of model to fit ('linear', 'logistic', 'poisson', 'multilevel').
    - group_var: str, specifies the grouping variable for multilevel models.
    
    Returns:
    - Fitted model object.
    """
    if model_type == 'linear':
        model = smf.ols(formula, data=data).fit()
    elif model_type == 'logistic':
        model = smf.logit(formula, data=data).fit()
    elif model_type == 'poisson':
        model = smf.glm(formula, data=data, family=sm.families.Poisson()).fit()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

def model_summary_to_dataframe(model):
    """
    Converts the model's summary to a pandas DataFrame.
    
    Parameters:
    - model: The fitted model object.
    
    Returns:
    - DataFrame containing the model summary.
    """
    summary_df = pd.DataFrame({
        "Coefficient": model.params,
        "Std Err": model.bse,
        "z or t": model.tvalues,
        "P>|z|": model.pvalues,
        "Conf. Interval Lower": model.conf_int()[0],
        "Conf. Interval Upper": model.conf_int()[1]
    }).reset_index()
    summary_df.columns = ['Term', 'Coefficient', 'Std Err', 'z or t', 'P>|z|', 'Conf. Interval Lower', 'Conf. Interval Upper']
    return summary_df

def generate_interpretation(model_summary, model_type):
    """
    Generates a detailed interpretation of the model based on its summary.
    
    Parameters:
    - model_summary: DataFrame containing the model's summary statistics.
    - model_type: Type of the model ('logistic', 'linear', 'poisson', 'multilevel').
    
    Returns:
    - A string containing the detailed interpretation of the model.
    """
    interpretation = f"Interpretation of the {model_type.title()} Model:\n"
    significant_terms = model_summary[model_summary['P>|z|'] < 0.05]
    
    if significant_terms.empty:
        interpretation += "No predictors were found to be statistically significant at the 5% significance level."
        return interpretation
    
    for _, row in significant_terms.iterrows():
        term = row['Term']
        coeff = row['Coefficient']
        p_value = row['P>|z|']
        
        if model_type == 'logistic':
            # For logistic regression, coefficients represent the log odds.
            interpretation += f"- The coefficient for {term} is {coeff:.4f}, with a p-value of {p_value:.4f}. This suggests that "
            interpretation += "as the predictor increases by one unit, the log odds of the dependent variable being 1 "
            interpretation += f"{'increase' if coeff > 0 else 'decrease'}.\n"
        elif model_type == 'linear':
            # For linear regression, coefficients represent the change in the dependent variable.
            interpretation += f"- The coefficient for {term} is {coeff:.4f}, with a p-value of {p_value:.4f}. This implies that "
            interpretation += "as the predictor increases by one unit, the dependent variable "
            interpretation += f"{'increases' if coeff > 0 else 'decreases'} by {abs(coeff):.4f} units.\n"
        # Additional conditions can be added for 'poisson' and 'multilevel' with appropriate interpretations.
        elif model_type == 'poisson':
            interpretation += f"- The coefficient for {term} is {coeff:.4f}, with a p-value of {p_value:.4f}. "
            interpretation += "In the context of a Poisson regression, this indicates that for a one-unit increase in the predictor, "
            interpretation += f"the expected count of the dependent variable {'increases' if coeff > 0 else 'decreases'} by a factor of {np.exp(coeff):.4f}. "
            if coeff > 0:
                interpretation += "This means the event becomes more likely as the predictor increases.\n"
            else:
                interpretation += "This means the event becomes less likely as the predictor increases.\n"

    
    # Include overall model fit (e.g., R-squared for linear regression, Log-likelihood for logistic regression) if relevant.
    if 'R-squared' in model_summary.columns:
        r_squared = model_summary.loc['R-squared', 'Coefficient']
        interpretation += f"\nThe model explains {r_squared:.4f} of the variance in the dependent variable."
    
    return interpretation


def consolidate_model_summaries(model_summaries):
    """
    Consolidates model summaries into a single dataframe, ensuring the model type 
    is presented as the first row of each section without introducing NaN values.
    
    Parameters:
    - model_summaries: List of tuples, where each tuple contains the model type as a string
                       and its corresponding summary as a pandas DataFrame.
                       
    Returns:
    - DataFrame containing all model summaries with the model type as the first row of each section,
      leaving cells empty where the model type row would introduce NaN values.
    """
    all_rows = []
    for model_type, df in model_summaries:
        # Initialize a row with empty strings for all columns except 'Term'
        empty_row = {col: "" for col in df.columns}
        empty_row['Term'] = model_type  # Set the 'Term' column to contain the model type
        
        # Create a DataFrame for the model type row, ensuring it has the same columns as the summary DataFrame
        model_type_row = pd.DataFrame([empty_row])
        
        # Concatenate the model type row with the actual model summary
        combined_df = pd.concat([model_type_row, df], ignore_index=True)
        
        all_rows.append(combined_df)
    
        consolidated_df = pd.concat(all_rows, ignore_index=True)
    
    # Optionally, you might want to replace NaN values across the entire DataFrame, if any remain
    consolidated_df.fillna("", inplace=True)
    
    return consolidated_df


def consolidate_model_interpretations(interpretations):
    """
    Consolidates model interpretations into a single dataframe.
    
    Parameters:
    - interpretations: List of tuples, where each tuple contains the model type as a string
                       and its corresponding interpretation as a string.
                       
    Returns:
    - DataFrame containing all model interpretations, each introduced by its model type.
    """
    # Initialize a list to store the DataFrames
    all_rows = []
    

    for model_type, interpretation in interpretations:
        # Create a DataFrame for each interpretation
        interpretation_df = pd.DataFrame({
            'Model Type': [model_type],
            'Interpretation': [interpretation]
        })
        
        # Append the DataFrame to the list
        all_rows.append(interpretation_df)

        # Concatenate all the mini DataFrames into a single DataFrame
        consolidated_df = pd.concat(all_rows, ignore_index=True)
        
    consolidated_df['Interpretation'] = consolidated_df['Interpretation'].str.replace("\n", " ", regex=False)

    return consolidated_df



# Extend the existing advanced_modeling function

def advanced_modeling(data, group_column, value_column, control_group):
    """
    Perform A/B testing with advanced modeling based on the nature of the dependent variable (binary or continuous).
    Generates a detailed report including model summaries and interpretations.
    """
    assert control_group in data[group_column].unique(), "Control group not found in group column."
    
    # Fix applied here: Convert to float and then check if every value is an integer
    dependent_var_type = 'binary' if data[value_column].dropna().astype(float).apply(lambda x: x.is_integer()).all() and data[value_column].nunique() == 2 else 'continuous'
    
    model_summaries = []
    interpretations = []
    
    if dependent_var_type == 'binary':
        # Logistic regression for binary outcomes
        logistic_formula = f"{value_column} ~ C({group_column})"
        logistic_model = fit_advanced_model(data, logistic_formula, model_type='logistic')
        logistic_summary = model_summary_to_dataframe(logistic_model)
        logistic_interpretation = generate_interpretation(logistic_summary, 'logistic')
        
        model_summaries.append(('Logistic Regression Model', logistic_summary))
        interpretations.append(('Logistic Regression Model Interpretation', logistic_interpretation))
        
        
    elif dependent_var_type == 'continuous':
        # Linear regression for continuous outcomes
        linear_formula = f"{value_column} ~ C({group_column})"
        linear_model = fit_advanced_model(data, linear_formula, model_type='linear')
        linear_summary = model_summary_to_dataframe(linear_model)
        linear_interpretation = generate_interpretation(linear_summary, 'linear')
        
        model_summaries.append(('Linear Regression Model', linear_summary))
        interpretations.append(('Linear Regression Model Interpretation', linear_interpretation))
        
        # Poisson regression might be considered for count data, depending on the context
        if data[value_column].min() >= 0:
            poisson_formula = f"{value_column} ~ C({group_column})"
            poisson_model = fit_advanced_model(data, poisson_formula, model_type='poisson')
            poisson_summary = model_summary_to_dataframe(poisson_model)
            poisson_interpretation = generate_interpretation(poisson_summary, 'poisson')
            
            model_summaries.append(('Poisson Regression Model', poisson_summary))
            interpretations.append(('Poisson Regression Model Interpretation', poisson_interpretation))

    else:
        raise ValueError("The dependent variable type could not be determined as either 'binary' or 'continuous'.")

    # Consolidate model summaries and interpretations for reporting
    model_summaries_df = consolidate_model_summaries(model_summaries)
    interpretations_df = consolidate_model_interpretations(interpretations)
                
    return model_summaries, interpretations, model_summaries_df, interpretations_df




### Main ###
def ab_test(data, group_column, value_column, control_group=None, alpha=0.05, handle_outliers=None, mc_correction=None):
    """Perform statistical analysis based on the groups in the data."""
    # Validate input data and columns
    validate_input(data, group_column, value_column)
    
    if handle_outliers is not None:
        assert len(handle_outliers) == 2, "handle_outliers must be a list with two elements: ['method', 'strategy']."
        data = handle_outliers_method(data, value_column, method=handle_outliers[0], strategy=handle_outliers[1])
  
    test_results = []
    p_values = []
    unique_groups = data[group_column].unique()
    
    if len(unique_groups) == 2:

        # Assess data characteristics
        group_a_data = data[data[group_column] == unique_groups[0]][value_column]
        group_b_data = data[data[group_column] == unique_groups[1]][value_column]
        n_a, n_b = len(group_a_data), len(group_b_data)
        normal_a = stats.shapiro(group_a_data).pvalue > 0.05
        normal_b = stats.shapiro(group_b_data).pvalue > 0.05
        small_sample = n_a < 30 or n_b < 30
        
        # Handling binary outcome data
        if data[value_column].nunique() == 2:
            success_a, success_b = sum(group_a_data), sum(group_b_data)
            contingency_table = pd.crosstab(data[group_column], data[value_column])

            # Fisher's Exact Test for small samples
            if small_sample:
                oddsratio, p_value = stats.fisher_exact(contingency_table)
                # Check if the null hypothesis is rejected
                if p_value < alpha:
                    successes += 1
                power = successes / 10000
                #interpretation = interpret_test_results(p_value, oddsratio, "Fisher's Exact Test", effect_size=oddsratio)
                test_results.append({'Test': "Fisher's Exact Test", 
                                     'P Value': p_value, 
                                     'Effect Size': f"Odds Ratio: {oddsratio:.4f}",
                                     'Power': power, 
                                     'Interpretation': interpret_test_results(p_value, oddsratio, "Fisher's Exact Test", effect_size=oddsratio, power=power)
                                     
                            })
            
            # Chi-square Test for larger samples
            elif not small_sample:
                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                phi = np.sqrt(chi2 / n_a + n_b)  # Phi coefficient for 2x2 table
                chi_square_effect_size = np.sqrt(chi2 / (n_a + n_b))
                power = power_analysis_chi_square(chi_square_effect_size, n_a + n_b)
                test_results.append({'Test': "Chi-square Test", 
                                     'P Value': p_value, 
                                     'Effect Size': f"Phi Coefficient: {phi:.4f}", 
                                     'Power': power,
                                     'Interpretation': interpret_test_results(p_value, chi2, "Chi-square Test", effect_size=phi, power=power)
                                     
                            })
            
            # Z-test for proportions for larger samples
            elif not small_sample and equal_variances and (normal_a and normal_b):
                count = np.array([success_a, success_b])
                nobs = np.array([n_a, n_b])
                z_stat, p_value = proportions_ztest(count, nobs)
                prop_a, prop_b = success_a / n_a, success_b / n_b
                cohens_h = cohen_h(prop_a, prop_b)
                power = power_analysis_ztest(prop_a, prop_b, n_a, n_b)
                test_results.append({'Test': 'Z-test for Proportions', 
                                     'P Value': p_value, 
                                     'Effect Size': f"Cohen's h: {cohens_h:.4f}", 
                                     'Power': power,
                                     'Interpretation': interpret_test_results(p_value, z_stat, "Z-test for Proportions", effect_size=cohens_h, power=power)
                                     

                            })
            

        # Handling continuous outcome data
        else:
            # Assess data characteristics
            normal_a, normal_b = stats.shapiro(group_a_data).pvalue > 0.05, stats.shapiro(group_b_data).pvalue > 0.05
            equal_variances = stats.levene(group_a_data, group_b_data).pvalue > 0.05
            
            # Welch's t-test for unequal variances or normal distributions
            if not small_sample and not equal_variances:
                t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data, equal_var=False)
                cohen_d_val = cohen_d(group_a_data, group_b_data)
                power = power_analysis_ttest(cohen_d_val, n_a, n_b)
                test_results.append({'Test': "Welch's t-test", 
                                     'P Value': p_value, 
                                     'Effect Size': f"Cohen's d: {cohen_d_val:.4f}",  
                                     'Power': power,
                                     'Interpretation': interpret_test_results(p_value, t_stat, "Welch's t-test", effect_size=cohen_d_val, power=power)
                            
                            })

            # Mann-Whitney U Test as a non-parametric alternative for small samples or non-normal distributions
            elif small_sample or not (normal_a and normal_b):
                u_stat, p_value = stats.mannwhitneyu(group_a_data, group_b_data, alternative='two-sided')
                n = n_a + n_b
                r_effect = r_effect_size(u_stat, n)  # Calculating r for Mann-Whitney
                power = power_analysis_mann_whitney(u_stat, n_a, n_b)
                test_results.append({'Test': 'Mann-Whitney U Test', 
                                     'P Value': p_value, 
                                     'Effect Size': f"r: {r_effect:.4f}", 
                                     'Power': power,
                                     'Interpretation': interpret_test_results(p_value, u_stat, "Mann-Whitney U Test", effect_size=r_effect, power=power)
                                     
                                     
                            })

            # Student's T-Test for large samples and equal variances
            elif not small_sample and equal_variances:
                # Assuming 'group_a' and 'group_b' are your continuous data groups
                mean_a, mean_b = np.mean(group_a_data), np.mean(group_b_data)
                std_a, std_b = np.std(group_a_data, ddof=1), np.std(group_b_data, ddof=1)  # Sample standard deviation
                n_a, n_b = len(group_a_data), len(group_b_data)  # Sample sizes
                
                # Perform the Student's t-test for independent samples
                t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data, equal_var=True)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2))
                cohens_t = (mean_a - mean_b) / pooled_std
                
                # Power analysis
                analysis = TTestIndPower()
                power = analysis.solve_power(effect_size=cohens_t, nobs1=n_a, ratio=n_b/n_a, alpha=alpha)
                
                # Append the results to your test_results list
                test_results.append({
                    'Test': "Student's t-test",
                    'P Value': p_value,
                    'Effect Size': f"Cohen's d: {cohens_t:.4f}",
                    'Power': power,
                    'Interpretation': interpret_test_results(p_value, t_stat, "Student's t-test", effect_size=cohens_t, power=power)
                    
                })


    elif len(unique_groups) > 2:
        # Preparing data for ANOVA or Kruskal-Wallis without unnecessary transformation
        grouped_data = [data[data[group_column] == group][value_column] for group in unique_groups]
        
        # Check normality for each group
        normality = all(stats.shapiro(group).pvalue > 0.05 for group in grouped_data)
        # Check equality of variances
        equal_variances = stats.levene(*grouped_data).pvalue > 0.05
            
        if normality and equal_variances:
                model = ols(f'{value_column} ~ C({group_column})', data=data).fit()
                anova_result = sm.stats.anova_lm(model, typ=2)
                test_statistic = anova_result['F'][0]
                p_value = anova_result['PR(>F)'][0]

                # Calculate Eta Squared for effect size
                eta_squared = calculate_eta_squared(anova_result)
                # Prepare for power analysis
                k_groups = data[group_column].nunique()
                n_total = len(data)
                power = calculate_power(effect_size=eta_squared, alpha=alpha, k_groups=k_groups, n_per_group=n_total)
                interpretation = interpret_test_results(p_value, test_statistic, 'ANOVA', effect_size=eta_squared, power=power)
                
                # Initial ANOVA test result
                test_results.append({
                    'Test': 'ANOVA',
                    'Test Statistic': test_statistic,
                    'P Value': p_value,
                    'Effect Size': eta_squared,
                    'Power': power,
                    'Interpretation': interpretation
                })
                
                if control_group and p_value < alpha:
                    mc = MultiComparison(data[value_column], data[group_column])
                    tukey_result = mc.tukeyhsd()
                    # Iterate over Tukey's results and append only comparisons involving the control group
                    for res in tukey_result.summary().data[1:]:  # Skipping header row
                        group1, group2, _, _, p_value, _ = res
                        if control_group in [group1, group2]:
                            comparison = f'{group1} vs. {group2}'
                            # Append each relevant post-hoc comparison
                            test_results.append({
                                'Post-hoc Test': 'Tukey',
                                'Groups Compared': comparison,
                                'P Value': p_value,
                                'Interpretation': interpret_test_results(p_value, test_statistic, 'Tukey')
                            })
                        
        else:
            # Preparing for Kruskal-Wallis test and power analysis
            distributions = [(data[data[group_column] == group][value_column].count(),
                            data[data[group_column] == group][value_column].mean(),
                            data[data[group_column] == group][value_column].std()) for group in unique_groups]

            # Simulate power for Kruskal-Wallis
            power = simulate_kruskal_wallis_power(distributions, alpha=alpha, num_simulations=1000)
 
            # Perform Kruskal-Wallis test and calculate effect size
            stat, p_value, eta_squared = perform_kruskal_wallis_test_and_effect_size(data, group_column, value_column)
            
            # Interpret the results (assuming interpret_test_results is already defined and updated)
            interpretation = interpret_test_results(p_value, stat, "Kruskal-Wallis", effect_size=eta_squared, power=power)

            #kruskal_statistic, kruskal_pvalue = kruskal(*grouped_data)
            #interpretation = interpret_test_results(kruskal_pvalue, kruskal_statistic, 'Kruskal-Wallis')
            test_results.append({
                'Test': 'Kruskal-Wallis',
                'Test Statistic': stat,
                'P Value': p_value,
                'Effect Size': eta_squared,
                'Power': power,
                'Interpretation': interpretation
            })
            
            if control_group and p_value < alpha:
                control_data = data[data[group_column] == control_group][value_column]
                for group in unique_groups:
                    if group != control_group:
                        group_data = data[data[group_column] == group][value_column]
                        mw_statistic, mw_pvalue = mannwhitneyu(control_data, group_data, alternative='two-sided')
                        interpretation = interpret_test_results(mw_pvalue, mw_statistic, 'Mann-Whitney U')
                        test_results.append({
                            'Post-hoc Test': 'Mann-Whitney U',
                            'Groups Compared': f'{control_group} vs. {group}',
                            'Test Statistic': mw_statistic,
                            'P Value': mw_pvalue,
                            'Interpretation': interpretation
                        })

    else:
        raise ValueError("group_column must contain at least two unique groups for analysis.")
        # Apply correction for multiple comparisons if specified
    
    # Apply multiple comparisons correction if specified
    if mc_correction and p_values:
        if mc_correction == 'bonferroni':
            corrected_p = multipletests(p_values, method='bonferroni')[1]
        elif mc_correction == 'fdr':
            corrected_p = multipletests(p_values, method='fdr_bh')[1]
        elif mc_correction == 'holm':
            corrected_p = multipletests(p_values, method='holm')[1]
        else:
            raise ValueError(f"Unsupported multiple comparisons correction method: {mc_correction}")
        
        for i, result in enumerate(test_results):
            result['Corrected P Value'] = corrected_p[i]
            result['Corrected Interpretation'] = interpret_test_results(corrected_p[i], alpha)
    
    df_results = pd.DataFrame(test_results)
    return df_results
