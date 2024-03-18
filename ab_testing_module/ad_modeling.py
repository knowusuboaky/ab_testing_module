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


