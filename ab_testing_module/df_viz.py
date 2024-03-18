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

