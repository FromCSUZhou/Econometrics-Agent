from __future__ import annotations
import sys
sys.path.append("/home/tianyang/ChatInterpreter/ML_Assistant")
from metagpt.tools.tool_registry import register_tool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import PanelOLS
import scipy.stats
from econml.dml import LinearDML
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

#%%

@register_tool(tags=["econometric algorithm"])
def ordinary_least_square_regression(dependent_variable, treatment_variable, covariate_variables, weights = None, cov_info = "nonrobust", target_type = "final_model", output_tables = False):
    
    """
    Use Ordinary Least Square Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable.
    The estimated ATE is the parameter of the treatment variable in the regression model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    If user specifies any fixed effect variable, in the OLS method this variable MUST BE transformed into dummy variables first (with one of the categories dropped to avoid multicollinearity with the constant term) and added into covariates.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        weights (pd.Series or None): Weights for data samples. If user does not specify any weight specification, the methodology will be standard OLS and this input should be None. If user specifies sample weights, the model will become Weighted Least Squares (WLS), a generalized version of OLS. 
        cov_info (str or dict): The covariance estimator used in the results. Four covariance estimators are supported: If no adjustment, input "nonrobust"; If heteroskedasticity-consistent adjustment (allows "HC0", "HC1", "HC2", "HC3"), take "HC0" as example, input "HC0", and if user specifies to use "robust" standard errors, input "HC1"; If heteroskedasticity and autocorrelation consistent adjustment (HAC) with integer lag terms, take maxlags equal to 5 for example, input {"HAV": 5}; If cluster adjustment with the target groups variable named "groups" (pd.Series or pd.dataframe), input {"cluster": groups}.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check Input
    if type(cov_info) == str and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")
    elif type(cov_info) == dict and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")

    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    # Run the regression
    if covariate_variables is None:
        X = treatment_variable
    else:
        X = pd.concat([treatment_variable, covariate_variables], axis = 1).astype(float)
    if weights is None:
        if type(cov_info) == str:
            regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type = cov_info)
        elif list(cov_info.keys())[0] == "HAC":
            regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
        elif list(cov_info.keys())[0] == "cluster":
            regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})
    else:
        if type(cov_info) == str:
            regression = sm.WLS(dependent_variable, sm.add_constant(X), weights = weights).fit(cov_type = cov_info)
        elif list(cov_info.keys())[0] == "HAC":
            regression = sm.WLS(dependent_variable, sm.add_constant(X), weights = weights).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
        elif list(cov_info.keys())[0] == "cluster":
            regression = sm.WLS(dependent_variable, sm.add_constant(X), weights = weights).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})

    # Output the table if required
    print("Estimated ATE: ", regression.params[treatment_variable.name])
    if output_tables is True:
        print(regression.summary())

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return regression.rsquared_adj
    elif target_type == "final_model":
        return regression
    
#%%

@register_tool(tags=["econometric algorithm"])
def propensity_score_construction(treatment_variable, covariate_variables):
    
    """
    Construct propensity score for each sample to receive binary treatment based on covariate variables, using binary Logistic regression.
    
    Args:
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable (1 for treatment, 0 for control).
        covariate_variables (pd.DataFrame): A dataframe of covariate variables, which should not contain nan value or intercept.

    Returns:
        pd.Series: The estimated propensity score for each sample, which will be named "propensity_score".
    """
    
    # Adjust input type
    treatment_variable = treatment_variable.astype(float)
    covariate_variables = covariate_variables.astype(float)
    
    # Directly apply Logistic regression method to estimate the propensity score
    clf = sm.Logit(treatment_variable, sm.add_constant(covariate_variables).astype(float)).fit()
    result_series = pd.Series(clf.predict(sm.add_constant(covariate_variables).astype(float)), index = covariate_variables.index)
    result_series.name = "propensity_score"
    return result_series

@register_tool(tags=["econometric algorithm"])
def propensity_score_visualize_propensity_score_distribution(treatment_variable, propensity_score):
    
    '''
    VISUALIZE propensity score distribution for treatment group and control group and compare their distributions.
    The ideal result is that treatment group and control group should distribute similarly across propensity score. One common scenario is
    treatment group has most sample with propensity score close to 1 and control group has most sample with propensity score close to 0.
    In this scenario, the best solution is to trim samples with extreme propensity scores and obtain a subsample with similarly distributed propensity score.
    
    Args:
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable (1 for treatment, 0 for control).
        propensity_score (pd.Series): Propensity score for each sample to receive treatment, which should not contain nan value.
    '''
    
    # Obtain treatment group propensity score and control group propensity score respectively
    treatment_group_propensity = propensity_score.loc[treatment_variable[treatment_variable == 1].index]
    control_group_propensity = propensity_score.loc[treatment_variable[treatment_variable == 0].index]
    
    # Visualize the distribution using histogram
    plt.hist(control_group_propensity, bins = 40, facecolor = "blue", edgecolor = "black", alpha = 0.7, label = "control")
    plt.hist(treatment_group_propensity, bins = 40, facecolor = "red", edgecolor = "black", alpha = 0.7, label = "treatment")

@register_tool(tags=["econometric algorithm"])
def propensity_score_matching(dependent_variable, treatment_variable, propensity_score, matched_num = 1, target_type = "ATE"):
    
    """
    Use propensity score matching method to estimate the Average Treatment Effect (ATE), or 
    Average Treatment Effect on the Treated (ATT) of the treatment variable towards the dependent variable. 
    This method is formally called Propensity Score Matching (PSM) approach.
    Note that the method allows sampling with replacement, as well as equal weighting when matched_num is larger than 1.
    The final return is the final estimated ATE or ATT as is required.
    Could refer to: https://www.stata.com/manuals/teteffectspsmatch.pdf
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable with no nan value (1 for treatment, 0 for control).
        propensity_score (pd.Series): Propensity score for each sample to receive treatment, which should not contain nan value.
        matched_num (int): The amount of nearest neighbors considered for each treatment entity. Should be an positive integer no smaller than 1.
        target_type (str): Target output type, which supports "ATE" and "ATT".
    """
    
    # Check inputs
    if target_type not in ["ATE", "ATT"]:
        raise RuntimeError("Target Type Input Not Supported! Only ATE or ATT could be supported!")
    
    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    
    # Process the ATE version
    if target_type == "ATE":
        
        # Match the entities and construct the matched control group dependent variable
        treatment_group_propensity_score_series = propensity_score.loc[treatment_variable[treatment_variable == 1].index]
        control_group_propensity_score_series = propensity_score.loc[treatment_variable[treatment_variable == 0].index]
        matched_control_dependent_variable_series = pd.Series(index = treatment_group_propensity_score_series.index)
        matched_treatment_dependent_variable_series = pd.Series(index = control_group_propensity_score_series.index)
        
        # Match the treatment group
        for each_index in treatment_group_propensity_score_series.index:
            selected_distance_metric = (control_group_propensity_score_series - treatment_group_propensity_score_series.loc[each_index]).map(lambda x: abs(x))
            selected_distance_metric = selected_distance_metric.sort_values()
            selected_index = selected_distance_metric.head(matched_num)
            selected_index = selected_distance_metric[selected_distance_metric.isin(list(selected_index.values))]
            matched_control_dependent_variable_series.loc[each_index] = dependent_variable.loc[selected_index.index].mean()
        
        # Match the control group
        for each_index in control_group_propensity_score_series.index:
            selected_distance_metric = (treatment_group_propensity_score_series - control_group_propensity_score_series.loc[each_index]).map(lambda x: abs(x))
            selected_distance_metric = selected_distance_metric.sort_values()
            selected_index = selected_distance_metric.head(matched_num)
            selected_index = selected_distance_metric[selected_distance_metric.isin(list(selected_index.values))]
            matched_treatment_dependent_variable_series.loc[each_index] = dependent_variable.loc[selected_index.index].mean()
    
        # Calculate the ATE and return the value
        ATE = pd.concat([dependent_variable.loc[treatment_group_propensity_score_series.index], matched_treatment_dependent_variable_series]).mean() - pd.concat([dependent_variable.loc[control_group_propensity_score_series.index], matched_control_dependent_variable_series]).mean()
        return ATE
    
    # Process the ATT version
    elif target_type == "ATT":
        
        # Match the entities and construct the matched control group dependent variable
        treatment_group_propensity_score_series = propensity_score.loc[treatment_variable[treatment_variable == 1].index]
        control_group_propensity_score_series = propensity_score.loc[treatment_variable[treatment_variable == 0].index]
        matched_dependent_variable_series = pd.Series(index = treatment_group_propensity_score_series.index)
        for each_index in treatment_group_propensity_score_series.index:
            selected_distance_metric = (control_group_propensity_score_series - treatment_group_propensity_score_series.loc[each_index]).map(lambda x: abs(x))
            selected_distance_metric = selected_distance_metric.sort_values()
            selected_index = selected_distance_metric.head(matched_num)
            selected_index = selected_distance_metric[selected_distance_metric.isin(list(selected_index.values))]
            matched_dependent_variable_series.loc[each_index] = dependent_variable.loc[selected_index.index].mean()
        
        # Calculate the ATT and return the value
        treatment_group_dependent_variable_series = dependent_variable.loc[treatment_group_propensity_score_series.index]
        ATT = treatment_group_dependent_variable_series.mean() - matched_dependent_variable_series.mean()
        return ATT

@register_tool(tags=["econometric algorithm"])
def propensity_score_inverse_probability_weighting(dependent_variable, treatment_variable, propensity_score, target_type = "ATE"):
    
    """
    Use propensity score inverse probability weighting (IPW) method to estimate the Average Treatment Effect (ATE), 
    or Average Treatment Effect on the Treated (ATT), of the treatment variable towards the dependent variable. 
    This method is formally called Propensity Score Inverse Probability Weighting (IPW) approach.
    The final return is the final estimated ATE or ATT as is required.
    Could refer to: hhttps://psantanna.com/Econ520/Slides/15-ipw/15slides.html#1

    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable with no nan value (1 for treatment, 0 for control).
        propensity_score (pd.Series): Propensity score for each sample to receive treatment, which should not contain nan value.
        target_type (str): Target output type, which supports "ATE" and "ATT".
    """
    
    # Check inputs
    if target_type not in ["ATE", "ATT"]:
        raise RuntimeError("Target Type Input Not Supported! Only ATE or ATT could be supported!")
    
    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    
    # Conduct calculation
    if target_type == "ATE":
        ATE_first_part = (dependent_variable * treatment_variable / propensity_score).sum() / (treatment_variable / propensity_score).sum()
        ATE_second_part = (dependent_variable * (1 - treatment_variable) / (1 - propensity_score)).sum() / ((1 - treatment_variable) / (1 - propensity_score)).sum()
        ATE = ATE_first_part - ATE_second_part
        return ATE
    else:
        ATT_first_part = (treatment_variable / treatment_variable.mean() * dependent_variable).mean()
        ATT_second_part_top = ((1 - treatment_variable) * propensity_score / (1 - propensity_score) * dependent_variable).mean()
        ATT_second_part_down = ((1 - treatment_variable) * propensity_score / (1 - propensity_score)).mean()
        return ATT_first_part - ATT_second_part_top / ATT_second_part_down

@register_tool(tags=["econometric algorithm"])
def propensity_score_regression(dependent_variable, treatment_variable, propensity_score, cov_type = None, target_type = "final_model", output_tables = False):
    
    """
    Use propensity score regression method to estimate Average Treatment Effect (ATE) of the treatment variable towards the dependent variable. 
    This method is formally called Regression Adjustment or Outcome Regression approach. Also, this is the homogeneous version.
    The estimated ATE is the parameter of the treatment variable in the regression model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    Note that the method is the homogeneous version.
    The final return is some clearly specified parameter or statistic within the regression model, or some regression model object within the function (by adjusting the argument input "target_type").
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable with no nan value (1 for treatment, 0 for control).
        propensity_score (pd.Series): Propensity score for each sample to receive treatment, which should not contain nan value.
        cov_type (str or None): The covariance estimator used in the results. If not specified by user, this could be None. NOTE THAT if user specifies to use "robust" standard errors, this input should be "HC1"!
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
        
    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    
    # Run the OLS regression
    if cov_type is None:
        OLS_model = sm.OLS(dependent_variable, sm.add_constant(pd.concat([treatment_variable, propensity_score], axis = 1))).fit()
    else:
        OLS_model = sm.OLS(dependent_variable, sm.add_constant(pd.concat([treatment_variable, propensity_score], axis = 1))).fit(cov_type = cov_type)

    # Output the table if required
    print("ATE Estimation: ", OLS_model.params[treatment_variable.name])
    if output_tables is True:
        print(OLS_model.summary())
    
    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -OLS_model.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return OLS_model.rsquared_adj
    elif target_type == "final_model":
        return OLS_model
    
@register_tool(tags=["econometric algorithm"])
def propensity_score_double_robust_estimator_augmented_IPW(dependent_variable, treatment_variable, propensity_score, covariate_variables, cov_type = None):
    
    """
    Use propensity score double robust augmented IPW method to estimate Average Treatment Effect (ATE) of the treatment variable towards the dependent variable. 
    This method is formally called Propensity Double Robust Estimator (Augmented IPW) approach. NOTE THAT this function is the DOUBLE ROBUST version!
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is the final estimated ATE.
    Can refer to: https://www.stata.com/manuals/teteffectsaipw.pdf
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable with no nan value (1 for treatment, 0 for control).
        covariate_variables (pd.DataFrame): A dataframe of covariate variables, which should not contain nan value or intercept.
        propensity_score (pd.Series): Propensity score for each sample to receive treatment, which should not contain nan value.
        cov_type (str or None): The covariance estimator used in the results. If not specified by user, this could be None. NOTE THAT if user specifies to use "robust" standard errors, this input should be "HC1"!
    """
    
    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    # Run the regression
    if covariate_variables is None:
        X = treatment_variable
    else:
        X = pd.concat([treatment_variable, covariate_variables], axis = 1).astype(float)
    if cov_type is None:
        regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit()
    else:
        regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type = cov_type)
    
    # Calculate the IPW
    IPW = treatment_variable / propensity_score + (1 - treatment_variable) / (1 - propensity_score)
    
    # Output the control group treatment version output
    selected_X = X.loc[treatment_variable[treatment_variable == 0].index]
    selected_X[treatment_variable.name] = 1
    control_group_constructed_output = pd.Series(regression.predict(sm.add_constant(selected_X, has_constant = "add")), index = selected_X.index)
    treatment_group_output = dependent_variable.loc[treatment_variable[treatment_variable == 1].index]
    
    # Calculate the ATE
    treatment_weighted = (treatment_group_output * IPW.loc[treatment_group_output.index] / IPW.loc[treatment_group_output.index].sum()).sum()
    control_weighted = (control_group_constructed_output * IPW.loc[control_group_constructed_output.index] / IPW.loc[control_group_constructed_output.index].sum()).sum()
    ATE = treatment_weighted - control_weighted
    return ATE
    
@register_tool(tags=["econometric algorithm"])
def propensity_score_double_robust_estimator_IPW_regression_adjustment(dependent_variable, treatment_variable, covariate_variables, propensity_score, cov_type = None, target_type = "final_model", output_tables = False):
    
    """
    Use propensity score double robust IPW regression adjustment method to estimate Average Treatment Effect (ATE) of the treatment variable towards the dependent variable. 
    This method is formally called Propensity Double Robust Estimator (IPW Regression Adjustment) approach. NOTE THAT this function is the DOUBLE ROBUST version!
    The ATE is the coefficient of the target treatment variable in the final OLS regression.
    The estimated ATE is the parameter of the treatment variable in the regression model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").
    Can refer to: https://www.stata.com/manuals/teteffectsipwra.pdf
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should be a binary variable with no nan value (1 for treatment, 0 for control).
        covariate_variables (pd.DataFrame): A dataframe of covariate variables, which should not contain nan value or intercept.
        propensity_score (pd.Series): Propensity score for each sample to receive treatment, which should not contain nan value.
        cov_type (str or None): The covariance estimator used in the results. If not specified by user, this could be None. NOTE THAT if user specifies to use "robust" standard errors, this input should be "HC1"!
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)
    
    # Calculate the IPW
    IPW = treatment_variable / propensity_score + (1 - treatment_variable) / (1 - propensity_score)
    IPW = IPW ** 0.5
        
    # Run the regression
    if covariate_variables is None:
        X = treatment_variable
    else:
        X = pd.concat([treatment_variable, covariate_variables], axis = 1).astype(float)
    if cov_type is None:
        regression = sm.WLS(dependent_variable, sm.add_constant(X), weights = IPW).fit()
    else:
        regression = sm.WLS(dependent_variable, sm.add_constant(X), weights = IPW).fit(cov_type = cov_type)
        
    # Output the table if required
    print("Estimated ATE: ", regression.params[treatment_variable.name])
    if output_tables is True:
        print(regression.summary())

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression
    
#%%

@register_tool(tags=["econometric algorithm"])
def IV_2SLS_regression(dependent_variable, treatment_variable, IV_variable, covariate_variables, cov_info = "nonrobust", target_type = "final_model", output_tables = False):
    
    """
    Use Instrument Variable - Two Step Least Square (IV-2SLS) method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, while ruling out endogeneiry in the original model.
    The estimated ATE is the parameter of the treatment variable in the second-step regression model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    If user specifies any fixed effect variable, in the IV-2SLS method this variable MUST BE transformed into dummy variables first (with one of the categories dropped to avoid multicollinearity with the constant term) and added into covariates.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").

    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should not contain nan value.
        IV_variable (pd.Series or pd.DataFrame): Proposed instrument variable(s). Could have only one or multiple IVs. Should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        cov_info (str or dict): The covariance estimator used in the results. Four covariance estimators are supported: If no adjustment, input "nonrobust"; If heteroskedasticity-consistent adjustment (allows "HC0", "HC1", "HC2", "HC3"), take "HC0" as example, input "HC0", and if user specifies to use "robust" standard errors, input "HC1"; If heteroskedasticity and autocorrelation consistent adjustment (HAC) with integer lag terms, take maxlags equal to 5 for example, input {"HAV": 5}; If cluster adjustment with the target groups variable named "groups" (pd.Series or pd.dataframe), input {"cluster": groups}.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final second-step regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check Input
    if type(cov_info) == str and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")
    elif type(cov_info) == dict and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")

    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    IV_variable = IV_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    # First step regression
    if covariate_variables is None:
        first_step_X = IV_variable
    else:
        first_step_X = pd.concat([IV_variable, covariate_variables], axis = 1).astype(float)
    if type(cov_info) == str:
        first_step_regression = sm.OLS(treatment_variable, sm.add_constant(first_step_X)).fit(cov_type = cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        first_step_regression = sm.OLS(treatment_variable, sm.add_constant(first_step_X)).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        first_step_regression = sm.OLS(treatment_variable, sm.add_constant(first_step_X)).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})
    predicted_treatment_result = pd.Series(first_step_regression.predict(sm.add_constant(first_step_X)), index = treatment_variable.index)
    predicted_treatment_result.name = treatment_variable.name

    # Second step regression
    if covariate_variables is None:
        second_step_X = predicted_treatment_result
    else:
        second_step_X = pd.concat([predicted_treatment_result, covariate_variables], axis = 1).astype(float)
    if type(cov_info) == str:
        second_step_regression = sm.OLS(dependent_variable, sm.add_constant(second_step_X)).fit(cov_type = cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        second_step_regression = sm.OLS(dependent_variable, sm.add_constant(second_step_X)).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        second_step_regression = sm.OLS(dependent_variable, sm.add_constant(second_step_X)).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})        

    # Output the table if required. ATE is the coefficient of the predicted treatment variable
    print("Estimated ATE: ", second_step_regression.params[predicted_treatment_result.name])
    if output_tables is True:
        print(second_step_regression.summary())

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -second_step_regression.pvalues[predicted_treatment_result.name]
    elif target_type == "rsquared":
        return second_step_regression.rsquared_adj
    elif target_type == "final_model":
        return second_step_regression
    
@register_tool(tags=["econometric algorithm"])
def IV_2SLS_IV_setting_test(dependent_variable, treatment_variable, IV_variable, covariate_variables, cov_type = None):

    """
    Test the fundamental assuptions that the proposed Instrument Variable should satisfy, which are:
        1. Relevant Condition: In the proposed population model, the proposed IV should be relevant with the target treatment variable;
        2. Exclusion Restriction: In the proposed population model, the proposed IV should not be relevant with the residual of this model.

    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        treatment_variable (pd.Series): Target treatment variable, which should not contain nan value.
        IV_variable (pd.Series): Proposed instrument variable. Could have ONLY ONE IV in this test function. Should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        cov_type (str or None): The covariance estimator used in the results. If not specified by user, this could be None.
    """

    # First test Relevant Condition. The coefficient of IV should be significant if it passes Relevant Condition requirement.
    if cov_type is None:
        relevant_test_OLS = sm.OLS(treatment_variable, sm.add_constant(IV_variable)).fit()
    else:
        relevant_test_OLS = sm.OLS(treatment_variable, sm.add_constant(IV_variable)).fit(cov_type = cov_type)
    print("Relevant Condition Test Result:")
    print(relevant_test_OLS.summary())
    
    # First test Exclusion Restriction. The coefficient of IV should be insignificant if it passes Exclusion Restriction requirement.
    if covariate_variables is None:
        restriction_test_X = treatment_variable
    else:
        restriction_test_X = pd.concat([treatment_variable, covariate_variables], axis = 1).astype(float)
    if cov_type is None:
        restriction_test_OLS = sm.OLS(dependent_variable, sm.add_constant(restriction_test_X)).fit()
    else:
        restriction_test_OLS = sm.OLS(dependent_variable, sm.add_constant(restriction_test_X)).fit(cov_type = cov_type)
    residual_series = pd.Series(restriction_test_OLS.resid, index = restriction_test_X.index)
    if cov_type is None:
        restriction_test_final_OLS = sm.OLS(residual_series, sm.add_constant(IV_variable)).fit()
    else:
        restriction_test_final_OLS = sm.OLS(residual_series, sm.add_constant(IV_variable)).fit(cov_type = cov_type)        
    print("Exclusion Restriction Test Result:")
    print(restriction_test_final_OLS.summary())
    
#%%

@register_tool(tags=["econometric algorithm"])
def Static_Diff_in_Diff_regression(dependent_variable, 
                                   treatment_entity_dummy, 
                                   treatment_finished_dummy, 
                                   covariate_variables, 
                                   entity_effect = False, 
                                   time_effect = False, 
                                   other_effect = None, 
                                   cov_type = "unadjusted", 
                                   target_type = "final_model", 
                                   output_tables = False):
    
    """
    Use Difference-in-Difference Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, in the PANEL DATA format. This is the STATIC version, 
    denoting that there is only one time spot when all entities in the treatment group is being treated. In other word, it's not the staggered method.
    The estimated ATE is the parameter of the interaction term (named "treatment_group_treated") in the regression model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value. The index of the series should be entity-time multi-index.
        treatment_entity_dummy (pd.Series): A dummy variables series denoting whether the entity is in the treatment group. This input should not contain nan value. The index of the series should be entity-time multi-index.
        treatment_finished_dummy (pd.Series): A dummy variables series denoting whether the treatment HAS BEEN implemented towards the treatment group. This input should not contain nan value. The index of the series should be entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        entity_effect (bool): Denote whether entity effect is included in the regression.
        time_effect (bool): Denote whether time effect is included in the regression.
        other_effect (pd.DataFrame or None): Denote whether other effects are included in the regression. If there are other effects required, this input should be a pd.DataFrame with the categorial variable column(s) and entity-time multi-index. If no other effects required, leave this input to be None.
        cov_type (str): The covariance estimator used in the results. Five covariance estimators are supported: "unadjusted" for homoskedastic residual, "robust" for heteroskedasticity control, "cluster_entity" for entity clustering, "cluster_time" for time clustering, and "cluster_both" for entity-time two-way clustering.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check if inputs are proper formatted
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'unadjusted', 'robust', 'cluster_entity', 'cluster_time' and 'cluster_both' as possible inputs!")
    count_effects = 0
    if entity_effect is True:
        count_effects += 1
    if time_effect is True:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed! Please note that now there are " + str(count_effects) + " effects in total!")

    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_entity_dummy = treatment_entity_dummy.astype(float)
    treatment_finished_dummy = treatment_finished_dummy.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    # Check to ensure dummy variables
    if list(treatment_entity_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("treatment_entity_dummy Input Error! Please Check!")
    if list(treatment_finished_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("treatment_finished_dummy Input Error! Please Check!")
        
    # Prepare the dataset
    treatment_entity_dummy.name = "treatment_group"
    treatment_finished_dummy.name = "treated"
    beta = treatment_entity_dummy * treatment_finished_dummy
    beta.name = "treatment_group_treated"
    if covariate_variables is None:
        X = pd.concat([beta, treatment_entity_dummy, treatment_finished_dummy], axis = 1)
    else:
        X = pd.concat([beta, treatment_entity_dummy, treatment_finished_dummy, covariate_variables], axis = 1).astype(float)
    if count_effects == 0:
        X = sm.add_constant(X)
    
    # Run the regression
    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_entity = True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_time = True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)

    # Output the table if required
    print("Estimated ATE: ", regression.params[beta.name])
    if output_tables is True:
        print(regression)

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues[beta.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression

@register_tool(tags=["econometric algorithm"])
def Staggered_Diff_in_Diff_regression(dependent_variable, 
                                      entity_treatment_dummy, 
                                      covariate_variables, 
                                      entity_effect = True, 
                                      time_effect = True, 
                                      other_effect = None, 
                                      cov_type = "unadjusted", 
                                      target_type = "final_model", 
                                      output_tables = False):
    
    """
    Use Difference-in-Difference Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, in the PANEL DATA format. This is the STAGGERED version, 
    denoting that there could be multiple time spot when different entities in the treatment group are being gradually treated. In other word, it's not the static method.
    The estimated ATE is the parameter of the entity treatment dummy in the regression model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value. The index of the series should be entity-time multi-index.
        entity_treatment_dummy (pd.Series): A dummy variables series denoting whether the treatment HAS BEEN implemented towards the entity in the timeslot. This input should not contain nan value. The index of the series should be entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        entity_effect (bool): Denote whether entity effect is included in the regression.
        time_effect (bool): Denote whether time effect is included in the regression.
        other_effect (pd.DataFrame or None): Denote whether other effects are included in the regression. If there are other effects required, this input should be a pd.DataFrame with the categorial variable column(s) and entity-time multi-index. If no other effects required, leave this input to be None.
        cov_type (str): The covariance estimator used in the results. Five covariance estimators are supported: "unadjusted" for homoskedastic residual, "robust" for heteroskedasticity control, "cluster_entity" for entity clustering, "cluster_time" for time clustering, and "cluster_both" for entity-time two-way clustering.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check if inputs are proper formatted
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'unadjusted', 'robust', 'cluster_entity', 'cluster_time' and 'cluster_both' as possible inputs!")
    count_effects = 0
    if entity_effect is True:
        count_effects += 1
    if time_effect is True:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed! Please note that now there are " + str(count_effects) + " effects in total!")

    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    # Check to ensure dummy variables
    if list(entity_treatment_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("entity_treatment_dummy Input Error! Please Check!")
        
    # Prepare the dataset
    entity_treatment_dummy.name = "treatment_entity_treated"
    if covariate_variables is None:
        X = entity_treatment_dummy
    else:
        X = pd.concat([entity_treatment_dummy, covariate_variables], axis = 1).astype(float)
    if count_effects == 0:
        X = sm.add_constant(X)
    
    # Run the regression
    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_entity = True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effect = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_time = True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effect = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)

    # Output the table if required
    print("Estimated ATE: ", regression.params[entity_treatment_dummy.name])
    if output_tables is True:
        print(regression)

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues[entity_treatment_dummy.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression

@register_tool(tags=["econometric algorithm"])
def Staggered_Diff_in_Diff_Event_Study_regression(dependent_variable, 
                                                  entity_treatment_dummy, 
                                                  covariate_variables, 
                                                  see_back_length: int = 4, 
                                                  see_forward_length: int = 3, 
                                                  entity_effect = True, 
                                                  time_effect = True, 
                                                  other_effect = None, 
                                                  cov_type = "unadjusted", 
                                                  target_type = "final_model", 
                                                  output_tables = False):

    """
    Use Difference-in-Difference Regression method to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable, in the PANEL DATA format. This is the STAGGERED version, 
    denoting that there could be multiple time spot when different entities in the treatment group are being gradually treated. In other word, it's not the static method.
    Also, this is the event study version, denoting that there should be enough amount of different treatment implementation time spots (more than see_back_length + see_forward_length).
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value. The index of the series should be entity-time multi-index.
        entity_treatment_dummy (pd.Series): A dummy variables series denoting whether the treatment HAS BEEN implemented towards the entity in the timeslot. This input should not contain nan value. The index of the series should be entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        entity_effect (bool): Denote whether entity effect is included in the regression.
        see_back_length (int): A positive int denote the length of see-back observation. 
        see_forward_length (int): A positive int denote the length of see-forward observation. 
        time_effect (bool): Denote whether time effect is included in the regression.
        other_effect (pd.DataFrame or None): Denote whether other effects are included in the regression. If there are other effects required, this input should be a pd.DataFrame with the categorial variable column(s) and entity-time multi-index. If no other effects required, leave this input to be None.
        cov_type (str): The covariance estimator used in the results. Five covariance estimators are supported: "unadjusted" for homoskedastic residual, "robust" for heteroskedasticity control, "cluster_entity" for entity clustering, "cluster_time" for time clustering, and "cluster_both" for entity-time two-way clustering.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check if inputs are proper formatted
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'unadjusted', 'robust', 'cluster_entity', 'cluster_time' and 'cluster_both' as possible inputs!")
    count_effects = 0
    if entity_effect is True:
        count_effects += 1
    if time_effect is True:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed! Please note that now there are " + str(count_effects) + " effects in total!")

    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    # Check to ensure dummy variables are proper formatted
    if list(entity_treatment_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("entity_treatment_dummy Input Error! Please Check!")
        
    # =========================================================================
        
    # Construct the event study variables
    entity_index_name, time_index_name = entity_treatment_dummy.index.names[0], entity_treatment_dummy.index.names[1]
    treatment_name = entity_treatment_dummy.name
    data_df = entity_treatment_dummy.reset_index()
    all_entity_list = list(data_df[entity_index_name].unique())
    all_time_list = list(data_df[time_index_name].unique())
    all_time_list.sort()
    policy_implementation_time_record_list = []
    for each_entity in all_entity_list:
        temp_df = data_df[data_df[entity_index_name] == each_entity]
        temp_df = temp_df.sort_values(by = time_index_name)
        if temp_df[temp_df[treatment_name] == 0].shape[0] == 0:
            raise RuntimeError("Entity:", each_entity, "was keeping implementing the policy since the start of this sample! Should consider to drop this entity!")
            # continue
        check_series = temp_df[treatment_name] - temp_df[treatment_name].shift().fillna(0)
        if check_series[check_series == 1].shape[0] == 0:
            # print("Entity:", each_entity, "did not implement this policy!")
            continue
        policy_time = temp_df.loc[check_series[check_series == 1].index[0], time_index_name]
        # print("Entity:", each_entity, "implemented this policy in time:", policy_time)
        if policy_time not in policy_implementation_time_record_list:
            policy_implementation_time_record_list.append(policy_time)
        
    # Check to ensure event study settings are proper formatted
    if see_back_length < 4 or see_forward_length < 3:
        raise RuntimeError("See back day length or see forward day length too few! Please check!")
    elif see_back_length + see_forward_length >= len(all_time_list):
        raise RuntimeError("See back day length or see forward day length too large! Please check!")
    
    # Construct Lead-Lag Dummy Variables (set Lead_D1 as default)
    Lead_column_name_list = ["Lead_D" + str(see_back_length) + "+"]
    for i in np.arange(see_back_length - 1, 1, -1):
        Lead_column_name_list.append("Lead_D" + str(i))
    Lag_column_name_list = []
    for i in np.arange(1, see_forward_length, 1):
        Lag_column_name_list.append("Lag_D" + str(i))
    Lag_column_name_list.append("Lag_D" + str(see_forward_length) + "+")
    Lead_and_Lag_column_name_list = Lead_column_name_list + ["D0"] + Lag_column_name_list
    
    # Calculate the values for each dummy variable
    considered_data_df = data_df[[entity_index_name, time_index_name]]
    considered_data_df[Lead_and_Lag_column_name_list] = np.nan
    for each_entity in all_entity_list:
        temp_df = data_df[data_df[entity_index_name] == each_entity]
        check_series = temp_df[treatment_name] - temp_df[treatment_name].shift().fillna(0)
        if check_series[check_series == 1].shape[0] == 0:  # If the state never implement the policy, for this version no indicator is added. So do not consider it here.  # TODO
            considered_data_df.loc[considered_data_df[entity_index_name] == each_entity, Lead_and_Lag_column_name_list] = 0
            continue
        # If the state always implement the policy, for this version such case should already be removed. So do not consider it here.  # TODO
        policy_time_index = check_series[check_series == 1].index[0]
        for each_index in temp_df.index:
            corresponding_each_time = temp_df.loc[each_index, time_index_name]
            if each_index - policy_time_index <= -see_back_length:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "Lead_D" + str(see_back_length) + "+"] = 1
            elif each_index - policy_time_index > -see_back_length and each_index - policy_time_index < -1:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "Lead_D" + str(policy_time_index - each_index)] = 1
            elif each_index == policy_time_index:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "D0"] = 1
            elif each_index - policy_time_index > 0 and each_index - policy_time_index < see_forward_length:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "Lag_D" + str(each_index - policy_time_index)] = 1
            elif each_index - policy_time_index >= see_forward_length:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "Lag_D" + str(see_forward_length) + "+"] = 1
        considered_data_df.loc[considered_data_df[entity_index_name] == each_entity, Lead_and_Lag_column_name_list] = considered_data_df.loc[considered_data_df[entity_index_name] == each_entity, Lead_and_Lag_column_name_list].fillna(0)
    considered_data_df = considered_data_df.set_index([entity_index_name, time_index_name])
    
    # =========================================================================

    # Prepare the dataset
    if covariate_variables is None:
        X = considered_data_df
    else:
        X = pd.concat([considered_data_df, covariate_variables], axis = 1).astype(float)
    if count_effects == 0:
        X = sm.add_constant(X)
        
    # Run the regression
    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_entity = True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_time = True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effects = entity_effect, time_effects = time_effect, other_effects = other_effect, drop_absorbed = True).fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)

    # Output the table if required
    if output_tables is True:
        print(regression)

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -regression.pvalues["D0"]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression
    
@register_tool(tags=["econometric algorithm"])
def Staggered_Diff_in_Diff_Event_Study_visualization(regression_model, see_back_length: int = 4, see_forward_length: int = 3):
    
    '''
    Visualize the Staggered Difference-in-Difference Event Study result. Note that this function needs the regression result from the previously defined function
    "Staggered_Diff_in_Diff_Event_Study_regression()", and need to set the input parameters "see_back_length" and "see_forward_length" well matched with the regression result.
    
    Args:
        regression_model (linearmodels.PanelOLS): The regression model returned from the previously defined function "Staggered_Diff_in_Diff_Event_Study_regression()", with the input 'target_type == "final_model"'.
        see_back_length (int): A positive int denote the length of see-back observation. 
        see_forward_length (int): A positive int denote the length of see-forward observation. 
    '''
    
    # Construct Lead-Lag Dummy Variables (set Lead_D1 as default)
    Lead_column_name_list = ["Lead_D" + str(see_back_length) + "+"]
    for i in np.arange(see_back_length - 1, 1, -1):
        Lead_column_name_list.append("Lead_D" + str(i))
    Lag_column_name_list = []
    for i in np.arange(1, see_forward_length, 1):
        Lag_column_name_list.append("Lag_D" + str(i))
    Lag_column_name_list.append("Lag_D" + str(see_forward_length) + "+")
    Lead_and_Lag_column_name_list = Lead_column_name_list + ["D0"] + Lag_column_name_list
    
    # Output the graph
    plt.plot(regression_model.params[Lead_and_Lag_column_name_list])
    plt.xticks(list(range(len(Lead_and_Lag_column_name_list))), Lead_and_Lag_column_name_list)
    plt.ylabel("Estimated Coefficients")
    plt.axhline(y = 0, color = "g", linestyle = "--")
    plt.axvline(x = 2.5, color = "g", linestyle = "--")
    for each_x_count in range(len(Lead_and_Lag_column_name_list)):
        each_x = regression_model.conf_int().index[each_x_count]
        plt.plot([each_x_count - 1 - 0.1, each_x_count - 1 + 0.1], [regression_model.conf_int().loc[each_x, "lower"], regression_model.conf_int().loc[each_x, "lower"]], color = "#f44336")
        plt.plot([each_x_count - 1 - 0.1, each_x_count - 1 + 0.1], [regression_model.conf_int().loc[each_x, "upper"], regression_model.conf_int().loc[each_x, "upper"]], color = "#f44336")
        plt.plot([each_x, each_x], [regression_model.conf_int().loc[each_x, "lower"], regression_model.conf_int().loc[each_x, "upper"]], color = "#f44336")
        
#%%

@register_tool(tags=["econometric algorithm"])
def Sharp_Regression_Discontinuity_Design_regression(dependent_variable, 
                                                     entity_treatment_dummy, 
                                                     running_variable, 
                                                     covariate_variables, 
                                                     running_variable_cutoff, 
                                                     running_variable_bandwidth, 
                                                     kernel_choice = "uniform", 
                                                     cov_info = "nonrobust", 
                                                     target_type = "final_model", 
                                                     output_tables = False):
    
    """
    Use Sharp Regression Discontinuity Design (Sharp RDD) Local Linear Regression approach to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable. This is the Sharp version, denoting that entities with treatment variable above the cutoff 
    will receive the final treatment FOR SURE. In other word, it's not the Fuzzy method.
    If user specifies any fixed effect variable, in the Sharp RDD method this variable MUST BE transformed into dummy variables first (with one of the categories dropped to avoid multicollinearity with the constant term) and added into covariates.
    The estimated ATE is the parameter of the entity treatment dummy in the regression model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        entity_treatment_dummy (pd.Series): A dummy variables series denoting whether the treatment is implemented towards the entity. This input should not contain nan value.
        running_variable (pd.Series): Target running variable to determine the possibility for the entity to receive treatment, which should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        running_variable_cutoff (float): Denote the threshold of the treatment variable, above which the entity will have higher chance to receive the final treatment.
        running_variable_bandwidth (float or None): Denote the bandwidth to consider in this study. If use full sample (i.e., no bandwidth selection in the task), this should be None.
        kernel_choice (str): Denote the choice of kernel function used in this analysis. Default is "uniform" that gives equal weights to all samples in the dataset. Can also accept "triangle" and "Epanechnikov".
        cov_info (str or dict): The covariance estimator used in the results. Four covariance estimators are supported: If no adjustment, input "nonrobust"; If heteroskedasticity-consistent adjustment (allows "HC0", "HC1", "HC2", "HC3"), take "HC0" as example, input "HC0", and if user specifies to use "robust" standard errors, input "HC1"; If heteroskedasticity and autocorrelation consistent adjustment (HAC) with integer lag terms, take maxlags equal to 5 for example, input {"HAV": 5}; If cluster adjustment with the target groups variable named "groups" (pd.Series or pd.dataframe), input {"cluster": groups}.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """    
    
    # Check if inputs are proper formatted
    if kernel_choice not in ["uniform", "triangle", "Epanechnikov"]:
        raise RuntimeError("Kernel function choice currently only supports 'uniform', 'triangle' and 'Epanechnikov'!")
    if type(cov_info) == str and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")
    elif type(cov_info) == dict and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")
    if running_variable_bandwidth is not None and running_variable_bandwidth <= 0:
        raise RuntimeError("If consider running variable bandwidth, this input MUST BE LARGER THAN 0! PLEASE CHECK!")
    if running_variable[running_variable > running_variable_cutoff].shape[0] == 0 or running_variable[running_variable < running_variable_cutoff].shape[0] == 0:
        raise RuntimeError("Running variable cutoff is out of the range for all running variable values! PLEASE CHECK!")
        
    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    running_variable = running_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    # =========================================================================
    
    # Construct variables
    if running_variable_bandwidth is None:
        running_variable_bandwidth = max(running_variable.max() - running_variable_cutoff, running_variable_cutoff - running_variable.min())
    selected_running_variable = running_variable[(running_variable >= running_variable_cutoff - running_variable_bandwidth) & (running_variable <= running_variable_cutoff + running_variable_bandwidth)]
    dependent_variable = dependent_variable.loc[selected_running_variable.index]
    entity_treatment_dummy = entity_treatment_dummy.loc[selected_running_variable.index]
    if covariate_variables is not None:
        covariate_variables = covariate_variables.loc[selected_running_variable.index].astype(float)
    if type(cov_info) == dict and list(cov_info.keys())[0] == "cluster":
        cov_info["cluster"] = cov_info["cluster"].loc[selected_running_variable.index]
    demeaned_selected_running_variable = selected_running_variable - running_variable_cutoff
    demeaned_selected_running_variable.name = "demeaned_" + selected_running_variable.name
    demeaned_selected_running_interaction_variable = demeaned_selected_running_variable * entity_treatment_dummy
    demeaned_selected_running_interaction_variable.name = "demeaned_interaction_" + entity_treatment_dummy.name
    
    # Construct weightings
    if kernel_choice == "uniform":
        weight = pd.Series(index = selected_running_variable.index).fillna(1 / selected_running_variable.shape[0])
    elif kernel_choice == "triangle":
        weight =  1 - ((selected_running_variable - running_variable_cutoff) / running_variable_bandwidth).abs()
    elif kernel_choice == "Epanechnikov":
        weight = selected_running_variable.map(lambda x: 0.75 * (1 - np.abs(((x - running_variable_cutoff) / running_variable_bandwidth)) ** 2))

    # Construct formula and dataset
    if covariate_variables is not None:
        regression_formula = dependent_variable.name + " ~ " + entity_treatment_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name + " + " + " + ".join(list(covariate_variables.columns))
        complete_dataset = pd.concat([dependent_variable, entity_treatment_dummy, demeaned_selected_running_variable, demeaned_selected_running_interaction_variable, covariate_variables], axis = 1)
    else:
        regression_formula = dependent_variable.name + " ~ " + entity_treatment_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name
        complete_dataset = pd.concat([dependent_variable, entity_treatment_dummy, demeaned_selected_running_variable, demeaned_selected_running_interaction_variable], axis = 1)

    # =========================================================================

    # Run the regressions
    if type(cov_info) == str:
        model = smf.wls(regression_formula, complete_dataset, weights = weight).fit(cov_type = cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        model = smf.wls(regression_formula, complete_dataset, weights = weight).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        model = smf.wls(regression_formula, complete_dataset, weights = weight).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})
    
    # Output the table if required
    print("Sharp RD Estimator: ", model.params[entity_treatment_dummy.name])
    if output_tables is True:
        print(model.summary())

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -model.pvalues[entity_treatment_dummy.name]
    elif target_type == "rsquared":
        return model.rsquared
    elif target_type == "final_model":
        return model

@register_tool(tags=["econometric algorithm"])
def Fuzzy_Regression_Discontinuity_Design_regression(dependent_variable, 
                                                     entity_treatment_dummy, 
                                                     running_variable, 
                                                     covariate_variables, 
                                                     running_variable_cutoff, 
                                                     running_variable_bandwidth, 
                                                     kernel_choice = "uniform", 
                                                     cov_info = "nonrobust", 
                                                     target_type = "estimator", 
                                                     output_tables = False):
    
    """
    Use Two-step Fuzzy Regression Discontinuity Design (Fuzzy RDD) Local Linear Regression approach to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable. This is the Fuzzy version, denoting that there could be higher possibility, 
    but not for sure, for an entity with treatment variable above the cutoff to receive the final treatment. In other word, it's not the Sharp method.
    If user specifies any fixed effect variable, in the Fuzzy RDD method this variable MUST BE transformed into dummy variables first (with one of the categories dropped to avoid multicollinearity with the constant term) and added into covariates.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        entity_treatment_dummy (pd.Series): A dummy variables series denoting whether the treatment is implemented towards the entity. This input should not contain nan value.
        running_variable (pd.Series): Target running variable to determine the possibility for the entity to receive treatment, which should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        running_variable_cutoff (float): Denote the threshold of the treatment variable, above which the entity will have higher chance to receive the final treatment.
        running_variable_bandwidth (float or None): Denote the bandwidth to consider in this study. If use full sample (i.e., no bandwidth selection in the task), this should be None.
        kernel_choice (str): Denote the choice of kernel function used in this analysis. Default is "uniform" that gives equal weights to all samples in the dataset. Can also accept "triangle" and "Epanechnikov".
        cov_info (str or dict): The covariance estimator used in the results. Four covariance estimators are supported: If no adjustment, input "nonrobust"; If heteroskedasticity-consistent adjustment (allows "HC0", "HC1", "HC2", "HC3"), take "HC0" as example, input "HC0", and if user specifies to use "robust" standard errors, input "HC1"; If heteroskedasticity and autocorrelation consistent adjustment (HAC) with integer lag terms, take maxlags equal to 5 for example, input {"HAV": 5}; If cluster adjustment with the target groups variable named "groups" (pd.Series or pd.dataframe), input {"cluster": groups}.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, two possible inputs are supported: "estimator" for final Fuzzy RDD estimator towards the causal effect of the treatment variable, and "final_models" for the two-step regression models in a list, with the first one as the first-step model and the second one as the second-step model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check if inputs are proper formatted
    if kernel_choice not in ["uniform", "triangle", "Epanechnikov"]:
        raise RuntimeError("Kernel function choice currently only supports 'uniform', 'triangle' and 'Epanechnikov'!")
    if type(cov_info) == str and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")
    elif type(cov_info) == dict and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")
    if running_variable_bandwidth is not None and running_variable_bandwidth <= 0:
        raise RuntimeError("If consider running variable bandwidth, this input MUST BE LARGER THAN 0! PLEASE CHECK!")
    if running_variable[running_variable > running_variable_cutoff].shape[0] == 0 or running_variable[running_variable < running_variable_cutoff].shape[0] == 0:
        raise RuntimeError("Running variable cutoff is out of the range for all running variable values! PLEASE CHECK!")
    
    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    running_variable = running_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)
    
    # =========================================================================

    # Construct variables
    if running_variable_bandwidth is None:
        running_variable_bandwidth = max(running_variable.max() - running_variable_cutoff, running_variable_cutoff - running_variable.min())
    selected_running_variable = running_variable[(running_variable >= running_variable_cutoff - running_variable_bandwidth) & (running_variable <= running_variable_cutoff + running_variable_bandwidth)]
    dependent_variable = dependent_variable.loc[selected_running_variable.index]
    entity_treatment_dummy = entity_treatment_dummy.loc[selected_running_variable.index]
    if covariate_variables is not None:
        covariate_variables = covariate_variables.loc[selected_running_variable.index].astype(float)
    if type(cov_info) == dict and list(cov_info.keys())[0] == "cluster":
        cov_info["cluster"] = cov_info["cluster"].loc[selected_running_variable.index]
    should_be_treated_dummy = selected_running_variable.map(lambda x: 1 if x >= running_variable_cutoff else 0)
    should_be_treated_dummy.name = selected_running_variable.name + "_dummy"
    demeaned_selected_running_variable = selected_running_variable - running_variable_cutoff
    demeaned_selected_running_variable.name = "demeaned_" + selected_running_variable.name
    demeaned_selected_running_interaction_variable = demeaned_selected_running_variable * should_be_treated_dummy
    demeaned_selected_running_interaction_variable.name = "demeaned_interaction_" + selected_running_variable.name
    
    # Construct weightings
    if kernel_choice == "uniform":
        weight = pd.Series(index = selected_running_variable.index).fillna(1 / selected_running_variable.shape[0])
    elif kernel_choice == "triangle":
        weight =  1 - ((selected_running_variable - running_variable_cutoff) / running_variable_bandwidth).abs()
    elif kernel_choice == "Epanechnikov":
        weight = selected_running_variable.map(lambda x: 0.75 * (1 - np.abs(((x - running_variable_cutoff) / running_variable_bandwidth)) ** 2))

    # Construct formula and dataset
    if covariate_variables is not None:
        regression_formula_1 = dependent_variable.name + " ~ " + should_be_treated_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name + " + " + " + ".join(list(covariate_variables.columns))
        regression_formula_2 = entity_treatment_dummy.name + " ~ " + should_be_treated_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name + " + " + " + ".join(list(covariate_variables.columns))
        complete_dataset = pd.concat([dependent_variable, entity_treatment_dummy, should_be_treated_dummy, demeaned_selected_running_variable, demeaned_selected_running_interaction_variable, covariate_variables], axis = 1)
    else:
        regression_formula_1 = dependent_variable.name + " ~ " + should_be_treated_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name
        regression_formula_2 = entity_treatment_dummy.name + " ~ " + should_be_treated_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name
        complete_dataset = pd.concat([dependent_variable, entity_treatment_dummy, should_be_treated_dummy, demeaned_selected_running_variable, demeaned_selected_running_interaction_variable], axis = 1)

    # =========================================================================

    # Run the regressions
    if type(cov_info) == str:
        model_1 = smf.wls(regression_formula_1, complete_dataset, weights = weight).fit(cov_type = cov_info)
        model_2 = smf.wls(regression_formula_2, complete_dataset, weights = weight).fit(cov_type = cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        model_1 = smf.wls(regression_formula_1, complete_dataset, weights = weight).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
        model_2 = smf.wls(regression_formula_2, complete_dataset, weights = weight).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        model_1 = smf.wls(regression_formula_1, complete_dataset, weights = weight).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})
        model_2 = smf.wls(regression_formula_2, complete_dataset, weights = weight).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})
    
    # Output the table if required
    print("Fuzzy RD Estimator: ", model_1.params[should_be_treated_dummy.name] / model_2.params[should_be_treated_dummy.name])
    if output_tables is True:
        print(model_1.summary())
        print(model_2.summary())

    # Return evaluation metric if needed
    if target_type == "estimator":
        return model_1.params[should_be_treated_dummy.name] / model_2.params[should_be_treated_dummy.name]
    elif target_type == "final_models":
        return [model_1, model_2]

@register_tool(tags=["econometric algorithm"])
def Fuzzy_RDD_Global_Polynomial_Estimator_regression(dependent_variable, 
                                                     entity_treatment_dummy, 
                                                     running_variable, 
                                                     covariate_variables, 
                                                     running_variable_cutoff, 
                                                     max_order, 
                                                     kernel_choice = "uniform", 
                                                     cov_info = "nonrobust", 
                                                     target_type = "final_model", 
                                                     output_tables = False):
    
    """
    Use Two-step Fuzzy Regression Discontinuity Design (Fuzzy RDD) Global Polynomial Estimator approach to estimate Average Treatment Effect (ATE) of 
    the treatment variable towards the dependent variable. This is the Fuzzy version, denoting that there could be higher possibility, 
    but not for sure, for an entity with treatment variable above the cutoff to receive the final treatment. In other word, it's not the Sharp method.
    Also, this is the Global Polynomial Estimator approach, meaning that all samples will be included in the analysis and no bandwidth is required or allowed.
    If user specifies any fixed effect variable, in the Fuzzy RDD Global Polynomial Estimator method this variable MUST BE transformed into dummy variables first (with one of the categories dropped to avoid multicollinearity with the constant term) and added into covariates.
    The estimated ATE is the parameter of the entity treatment dummy in the second-step regression model.
    NOTE THAT THIS FUNCTION DOES NOT RETURN THE FINAL REGRESSION TABLE! All tables can (and only can) be printed out during the function.
    The final return is some clearly specified parameter or statistic within the regressions, or some regression model object within the function (by adjusting the argument input "target_type").
    
    Args:
        dependent_variable (pd.Series): Target dependent variable, which should not contain nan value.
        entity_treatment_dummy (pd.Series): A dummy variables series denoting whether the treatment is implemented towards the entity. This input should not contain nan value.
        running_variable (pd.Series): Target running variable to determine the possibility for the entity to receive treatment, which should not contain nan value.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables. If user does not specify any covariate variable, this could be None. Otherwise, it should not contain nan value.
        running_variable_cutoff (float): Denote the threshold of the treatment variable, above which the entity will have higher chance to receive the final treatment.
        max_order (int): Denote the highest polynomial order in the analysis. Should be an integer no smaller than 1.
        kernel_choice (str): Denote the choice of kernel function used in this analysis. Default is "uniform" that gives equal weights to all samples in the dataset. Can also accept "triangle" and "Epanechnikov".
        cov_info (str or dict): The covariance estimator used in the results. Four covariance estimators are supported: If no adjustment, input "nonrobust"; If heteroskedasticity-consistent adjustment (allows "HC0", "HC1", "HC2", "HC3"), take "HC0" as example, input "HC0", and if user specifies to use "robust" standard errors, input "HC1"; If heteroskedasticity and autocorrelation consistent adjustment (HAC) with integer lag terms, take maxlags equal to 5 for example, input {"HAV": 5}; If cluster adjustment with the target groups variable named "groups" (pd.Series or pd.dataframe), input {"cluster": groups}.
        target_type (str or None): Denote whether this function need to return any specific evaluation metric or any other content. If only want to print out regression tables, this should be None. Otherwise, three possible inputs are supported: "neg_pvalue" for the regression treatment variable coefficient p-value's negative value, "rsquared" for the adjusted R-squared value of the regression, and "final_model" for the final second-step regression model.
        output_tables (bool): Denote whether this function need to print out regression tables. If want to print out the tabels, this should be True. If only want the evaluation metric outputs, this should be False.
    """
    
    # Check if inputs are proper formatted
    if kernel_choice not in ["uniform", "triangle", "Epanechnikov"]:
        raise RuntimeError("Kernel function choice currently only supports 'uniform', 'triangle' and 'Epanechnikov'!")
    if type(cov_info) == str and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")
    elif type(cov_info) == dict and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported! This function supports 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' (with maxlags input) and 'cluster' (with target groups) as possible inputs!")
    if running_variable[running_variable > running_variable_cutoff].shape[0] == 0 or running_variable[running_variable < running_variable_cutoff].shape[0] == 0:
        raise RuntimeError("Running variable cutoff is out of the range for all running variable values! PLEASE CHECK!")
    if max_order < 1:
        raise RuntimeError("max_order input must be no smaller than 1! PLEASE CHECK!")
    
    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    running_variable = running_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)
    
    # =========================================================================
    
    # Construct variables
    should_be_treated_dummy = running_variable.map(lambda x: 1 if x >= running_variable_cutoff else 0)
    should_be_treated_dummy.name = running_variable.name + "_dummy"

    # Construct weightings
    running_variable_bandwidth = max(running_variable.max() - running_variable_cutoff, running_variable_cutoff - running_variable.min())
    if kernel_choice == "uniform":
        weight = pd.Series(index = running_variable.index).fillna(1 / running_variable.shape[0])
    elif kernel_choice == "triangle":
        weight =  1 - ((running_variable - running_variable_cutoff) / running_variable_bandwidth).abs()
    elif kernel_choice == "Epanechnikov":
        weight = running_variable.map(lambda x: 0.75 * (1 - np.abs(((x - running_variable_cutoff) / running_variable_bandwidth)) ** 2))

    # =========================================================================

    # Construct higher order terms for step 1 regression
    all_constructed_terms_list_step_1 = []
    for each_order in range(1, max_order + 1):
        demeaned_running_variable = (running_variable - running_variable_cutoff) ** each_order
        demeaned_running_variable.name = "demeaned_order_" + str(each_order) + "_" + running_variable.name
        demeaned_running_interaction_variable = demeaned_running_variable * should_be_treated_dummy
        demeaned_running_interaction_variable.name = "demeaned_order_" + str(each_order) + "_interaction_" + running_variable.name
        all_constructed_terms_list_step_1.append(demeaned_running_variable)
        all_constructed_terms_list_step_1.append(demeaned_running_interaction_variable)
    all_constructed_terms_list_step_1 = pd.concat(all_constructed_terms_list_step_1, axis = 1)
    all_constructed_terms_name_list_step_1 = list(all_constructed_terms_list_step_1.columns)
    
    # Construct formula and dataset for step 1 regression
    if covariate_variables is not None:
        regression_formula_1 = entity_treatment_dummy.name + " ~ " + should_be_treated_dummy.name + " + " + " + ".join(all_constructed_terms_name_list_step_1) + " + " + " + ".join(list(covariate_variables.columns))
        complete_dataset_1 = pd.concat([entity_treatment_dummy, should_be_treated_dummy, all_constructed_terms_list_step_1, covariate_variables.astype(float)], axis = 1)
    else:
        regression_formula_1 = entity_treatment_dummy.name + " ~ " + should_be_treated_dummy.name + " + " + " + ".join(all_constructed_terms_name_list_step_1)
        complete_dataset_1 = pd.concat([entity_treatment_dummy, should_be_treated_dummy, all_constructed_terms_list_step_1], axis = 1)
        
    # Run the step 1 regression and produce prediction
    if type(cov_info) == str:
        model_1 = smf.wls(regression_formula_1, complete_dataset_1, weights = weight).fit(cov_type = cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        model_1 = smf.wls(regression_formula_1, complete_dataset_1, weights = weight).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        model_1 = smf.wls(regression_formula_1, complete_dataset_1, weights = weight).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})
    entity_treatment_dummy_hat = pd.Series(model_1.predict(complete_dataset_1[complete_dataset_1.columns[1:]]))
    entity_treatment_dummy_hat.name = entity_treatment_dummy.name
    
    # =========================================================================
    
    # Construct higher order terms for step 2 regression
    all_constructed_terms_list_step_2 = []
    for each_order in range(1, max_order + 1):
        demeaned_running_variable = (running_variable - running_variable_cutoff) ** each_order
        demeaned_running_variable.name = "demeaned_order_" + str(each_order) + "_" + running_variable.name
        demeaned_running_interaction_variable = demeaned_running_variable * entity_treatment_dummy_hat
        demeaned_running_interaction_variable.name = "demeaned_order_" + str(each_order) + "_interaction_" + running_variable.name
        all_constructed_terms_list_step_2.append(demeaned_running_variable)
        all_constructed_terms_list_step_2.append(demeaned_running_interaction_variable)
    all_constructed_terms_list_step_2 = pd.concat(all_constructed_terms_list_step_2, axis = 1)
    all_constructed_terms_name_list_step_2 = list(all_constructed_terms_list_step_2.columns)

    # Construct formula and dataset for step 2 regression
    if covariate_variables is not None:
        regression_formula_2 = dependent_variable.name + " ~ " + entity_treatment_dummy_hat.name + " + " + " + ".join(all_constructed_terms_name_list_step_2) + " + " + " + ".join(list(covariate_variables.columns))
        complete_dataset_2 = pd.concat([dependent_variable, entity_treatment_dummy_hat, all_constructed_terms_list_step_2, covariate_variables.astype(float)], axis = 1)
    else:
        regression_formula_2 = dependent_variable.name + " ~ " + entity_treatment_dummy_hat.name + " + " + " + ".join(all_constructed_terms_name_list_step_2)
        complete_dataset_2 = pd.concat([dependent_variable, entity_treatment_dummy_hat, all_constructed_terms_list_step_2], axis = 1)
        
    # Run the step 2 regression and produce inference
    if type(cov_info) == str:
        model_2 = smf.wls(regression_formula_2, complete_dataset_2, weights = weight).fit(cov_type = cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        model_2 = smf.wls(regression_formula_2, complete_dataset_2, weights = weight).fit(cov_type = "HAC", cov_kwds = {"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        model_2 = smf.wls(regression_formula_2, complete_dataset_2, weights = weight).fit(cov_type = "cluster", cov_kwds = {"groups": cov_info["cluster"]})    

    # =========================================================================

    # Output the table if required
    print("Final ATE Estimation: ", model_2.params[entity_treatment_dummy_hat.name])
    if output_tables is True:
        print(model_1.summary())
        print(model_2.summary())

    # Return evaluation metric if needed
    if target_type == "neg_pvalue":
        return -model_2.pvalues[entity_treatment_dummy_hat.name]
    elif target_type == "rsquared":
        return model_2.rsquared
    elif target_type == "final_model":
        return model_2

#%%

@register_tool(tags=["econometric algorithm"])
def Linear_Double_Machine_Learning(dependent_variable,
                                   treatment_variable,
                                   covariate_variables,
                                   model_y=None,
                                   model_t=None,
                                   n_splits: int = 3,
                                   random_state: int = 0,
                                   target_type: str = "ATE"):

    """
    Use EconML's LinearDML to estimate Average Treatment Effect (ATE) via the
    double machine learning (DML) framework.

    Args:
        dependent_variable (pd.Series): Target dependent variable with no ``NaN`` value.
        treatment_variable (pd.Series): Target treatment variable with no ``NaN`` value.
        covariate_variables (pd.DataFrame or None): Covariate variables. If ``None``,
            no covariate adjustment will be performed.
        model_y (object or None): Machine learning model for the outcome. ``RandomForestRegressor``
            will be used by default.
        model_t (object or None): Machine learning model for the treatment. ``RandomForestRegressor``
            will be used by default.
        n_splits (int): Number of cross-fitting splits.
        random_state (int): Random seed used in the estimator.
        target_type (str): "ATE" to return the estimated average treatment effect or
            "final_model" to return the fitted ``LinearDML`` estimator.

    Returns:
        float or econml.dml.LinearDML: The estimated ATE or the fitted estimator.
    """

    # Adjust input type
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if model_y is None:
        model_y = RandomForestRegressor(n_estimators=100, random_state=random_state)
    if model_t is None:
        model_t = RandomForestRegressor(n_estimators=100, random_state=random_state)

    discrete_treat = set(treatment_variable.dropna().unique()) <= {0, 1}
    estimator = LinearDML(model_y=model_y,
                          model_t=model_t,
                          discrete_treatment=discrete_treat,
                          cv=n_splits,
                          random_state=random_state)

    estimator.fit(Y=dependent_variable,
                  T=treatment_variable,
                  X=covariate_variables)

    ate = estimator.ate(X=covariate_variables)
    print("Estimated ATE: ", ate)

    if target_type == "final_model":
        return estimator
    else:
        return ate
