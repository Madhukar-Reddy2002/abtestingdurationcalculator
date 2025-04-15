import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import math

# Set page config
st.set_page_config(
    page_title="A/B Test Significance Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #424242;
        margin-bottom: 1rem;
    }
    .results-summary {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-size: 16px;
        line-height: 1.6;
    }
    .results-summary strong {
        font-weight: 600;
    }
    .significant-result {
        color: #28a745;
    }
    .not-significant-result {
        color: #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        color: #424242;
        font-size: 26px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">A/B Test Significance Calculator</div>', unsafe_allow_html=True)
st.markdown("""
This calculator determines the **statistical significance** of your A/B test results and how much confidence you can have in the observed uplift.
""")

# Sidebar for configurations
with st.sidebar:
    st.markdown('<div class="sub-header">Test Configuration</div>', unsafe_allow_html=True)
    
    # Number of variants
    num_variants = st.number_input(
        "Number of Variants",
        min_value=2,
        max_value=5,
        value=2,
        step=1,
        help="Total number of variations in your test, including the control"
    )
    
    # Statistical settings
    st.markdown("#### Statistical Settings")
    
    hypothesis_type = st.radio(
        "Hypothesis Type",
        options=["Two-tailed", "One-tailed"],
        index=0,
        help="Two-tailed: Test if variant is different from control (better or worse)\nOne-tailed: Test if variant is better than control"
    )
    
    # Multiple comparison correction
    apply_correction = st.checkbox(
        "Apply Bonferroni Correction",
        value=True,
        help="Apply correction when comparing multiple variants to reduce false positives"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        metric_type = st.radio(
            "Metric Type",
            options=["Binary (Conversion)", "Continuous (Revenue/Value)"],
            index=0,
            help="Binary: Count-based metrics like clicks, conversions\nContinuous: Value-based metrics like revenue, time on page"
        )
        
        min_sample_warning = st.number_input(
            "Minimum Sample Size",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Show a warning if sample size is below this threshold"
        )

# Function to calculate statistical power based on sample size
def calculate_power(control_rate, variant_rate, control_size, variant_size, alpha=0.05):
    # Effect size in terms of proportions
    effect_size = abs(variant_rate - control_rate)
    
    # Calculate standard errors
    se_control = math.sqrt(control_rate * (1 - control_rate) / control_size)
    se_variant = math.sqrt(variant_rate * (1 - variant_rate) / variant_size)
    
    # Pooled standard error
    se_pooled = math.sqrt(se_control**2 + se_variant**2)
    
    # Z-score for the effect
    z_effect = effect_size / se_pooled if se_pooled > 0 else 0
    
    # Z-score for alpha (two-tailed)
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - z_effect)
    
    return power * 100  # Return as percentage

# Function to calculate statistical significance for binary (conversion) metrics
def analyze_binary_test(variants_data, alpha=0.05, one_tailed=False, correction=False):
    results = []
    control = variants_data[0]  # Assume first variant is control
    
    # Apply Bonferroni correction if needed
    if correction and len(variants_data) > 2:
        alpha = alpha / (len(variants_data) - 1)
    
    for i, variant in enumerate(variants_data):
        if i == 0:  # Skip comparison for control
            result = {
                "name": variant["name"],
                "visitors": variant["visitors"],
                "conversions": variant["conversions"],
                "conversion_rate": variant["conversions"] / variant["visitors"] if variant["visitors"] > 0 else 0,
                "is_control": True,
                "significant": False,
                "p_value": 1.0,
                "effect": 0,
                "relative_effect": 0,
                "confidence_interval": (0, 0),
                "confidence_level": 0,
                "power": 0
            }
            results.append(result)
            continue
        
        # Calculate p-value using proportion test
        control_rate = control["conversions"] / control["visitors"] if control["visitors"] > 0 else 0
        variant_rate = variant["conversions"] / variant["visitors"] if variant["visitors"] > 0 else 0
        
        # Calculate standard error
        se_control = math.sqrt(control_rate * (1 - control_rate) / control["visitors"]) if control["visitors"] > 0 else 0
        se_variant = math.sqrt(variant_rate * (1 - variant_rate) / variant["visitors"]) if variant["visitors"] > 0 else 0
        se_diff = math.sqrt(se_control**2 + se_variant**2)
        
        # Calculate z-score
        z_score = (variant_rate - control_rate) / se_diff if se_diff > 0 else 0
        
        # Calculate p-value based on hypothesis type
        if one_tailed:
            p_value = 1 - stats.norm.cdf(z_score) if z_score > 0 else 1.0
        else:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Calculate effect size
        absolute_effect = variant_rate - control_rate
        relative_effect = (variant_rate / control_rate - 1) * 100 if control_rate > 0 else 0
        
        # Calculate confidence interval
        ci_factor = stats.norm.ppf(1 - alpha/2)
        ci_lower = absolute_effect - ci_factor * se_diff
        ci_upper = absolute_effect + ci_factor * se_diff
        
        # Determine if result is significant
        significant = p_value < alpha
        
        # Calculate confidence level (1 - p_value)
        confidence_level = (1 - p_value) * 100
        
        # Calculate statistical power
        power = calculate_power(control_rate, variant_rate, control["visitors"], variant["visitors"], alpha)
        
        result = {
            "name": variant["name"],
            "visitors": variant["visitors"],
            "conversions": variant["conversions"],
            "conversion_rate": variant_rate,
            "is_control": False,
            "significant": significant,
            "p_value": p_value,
            "z_score": z_score,
            "effect": absolute_effect,
            "relative_effect": relative_effect,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": confidence_level,
            "power": power
        }
        results.append(result)
    
    return results

# Function to calculate statistical significance for continuous metrics
def analyze_continuous_test(variants_data, alpha=0.05, one_tailed=False, correction=False):
    results = []
    control = variants_data[0]  # Assume first variant is control
    
    # Apply Bonferroni correction if needed
    if correction and len(variants_data) > 2:
        alpha = alpha / (len(variants_data) - 1)
    
    for i, variant in enumerate(variants_data):
        if i == 0:  # Skip comparison for control
            result = {
                "name": variant["name"],
                "visitors": variant["visitors"],
                "mean": variant["mean"],
                "std_dev": variant["std_dev"],
                "is_control": True,
                "significant": False,
                "p_value": 1.0,
                "effect": 0,
                "relative_effect": 0,
                "confidence_interval": (0, 0),
                "confidence_level": 0,
                "power": 0
            }
            results.append(result)
            continue
        
        # Calculate degrees of freedom
        df = control["visitors"] + variant["visitors"] - 2
        
        # Calculate pooled standard deviation
        pooled_var = ((control["visitors"] - 1) * control["std_dev"]**2 + 
                      (variant["visitors"] - 1) * variant["std_dev"]**2) / df
        pooled_std = math.sqrt(pooled_var)
        
        # Calculate standard error of difference
        se_diff = pooled_std * math.sqrt(1/control["visitors"] + 1/variant["visitors"])
        
        # Calculate t-statistic
        t_stat = (variant["mean"] - control["mean"]) / se_diff if se_diff > 0 else 0
        
        # Calculate p-value based on hypothesis type
        if one_tailed:
            p_value = 1 - stats.t.cdf(t_stat, df) if t_stat > 0 else 1.0
        else:
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Calculate effect size
        absolute_effect = variant["mean"] - control["mean"]
        relative_effect = (variant["mean"] / control["mean"] - 1) * 100 if control["mean"] != 0 else 0
        
        # Calculate confidence interval
        ci_factor = stats.t.ppf(1 - alpha/2, df)
        ci_lower = absolute_effect - ci_factor * se_diff
        ci_upper = absolute_effect + ci_factor * se_diff
        
        # Determine if result is significant
        significant = p_value < alpha
        
        # Calculate confidence level (1 - p_value)
        confidence_level = (1 - p_value) * 100
        
        # Rough estimate of power (this is simplified)
        # For a proper power calculation, you'd need more information
        effect_size_d = abs(absolute_effect) / pooled_std
        ncp = effect_size_d * math.sqrt(control["visitors"] * variant["visitors"] / (control["visitors"] + variant["visitors"]))
        power = 1 - stats.nct.cdf(stats.t.ppf(1-alpha, df), df, ncp)
        power = power * 100  # Convert to percentage
        
        result = {
            "name": variant["name"],
            "visitors": variant["visitors"],
            "mean": variant["mean"],
            "std_dev": variant["std_dev"],
            "is_control": False,
            "significant": significant,
            "p_value": p_value,
            "t_stat": t_stat,
            "effect": absolute_effect,
            "relative_effect": relative_effect,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": confidence_level,
            "power": power
        }
        results.append(result)
    
    return results

# Function to generate conversational result summary
def generate_result_summary(result, control_name, metric_type="conversion"):
    variant_name = result["name"]
    effect = result["relative_effect"]
    confidence = result["confidence_level"]
    power = result["power"]
    significant = result["significant"]
    total_visitors = result["visitors"] + [r["visitors"] for r in results if r["name"] == control_name][0]
    
    # Format effect with sign and proper precision
    effect_abs = abs(effect)
    effect_sign = "better" if effect > 0 else "worse"
    
    # Generate summary text
    summary = f"**Your test results**\n\n"
    summary += f"**Test \"{variant_name}\" {metric_type}d {effect_abs:.1f}% {effect_sign}** than Test \"{control_name}\". "
    summary += f"I am **{confidence:.0f}%** certain that the changes in Test \"{variant_name}\" "
    
    if effect > 0:
        summary += f"will improve your {metric_type} rate. "
    else:
        summary += f"will decrease your {metric_type} rate. "
    
    if significant:
        summary += f"**Your results are statistically significant.**"
    else:
        summary += f"**Unfortunately, your results are not statistically significant.**"
    
    # Add information about the reliability based on power
    summary += f"\n\n**Test reliability: {power:.0f}%**. "
    
    if power < 50:
        summary += f"With only {total_visitors} total visitors, the sample size is too small to be confident in these results. "
        summary += f"Consider running the test longer to collect more data."
    elif power < 80:
        summary += f"With {total_visitors} total visitors, we have moderate confidence in these results. "
        summary += f"For higher reliability, consider collecting more data."
    else:
        summary += f"With {total_visitors} total visitors, this test has sufficient statistical power to detect the observed difference."
    
    return summary

# Create dynamic form for entering variant data
st.markdown('<div class="sub-header">Enter Test Results</div>', unsafe_allow_html=True)

# Create tabs for different metric types
tab_labels = ["Binary (Conversion)", "Continuous (Revenue/Value)"] if metric_type == "Continuous (Revenue/Value)" else ["Binary (Conversion)"]
tabs = st.tabs(tab_labels)

with tabs[0]:  # Binary metrics tab
    st.markdown("##### 📊 Enter visitor and conversion data for each variant")
    
    binary_variants = []
    
    # Create a more compact form layout
    for i in range(num_variants):
        variant_name = "Control" if i == 0 else f"Variant {i}"
        
        cols = st.columns([1.5, 1, 1])
        with cols[0]:
            name = st.text_input(f"Variant name", value=variant_name, key=f"name_bin_{i}")
        with cols[1]:
            visitors = st.number_input(f"Visitors", value=1000 if i == 0 else 1000, min_value=1, key=f"visitors_bin_{i}")
        with cols[2]:
            max_conv = visitors  # Can't have more conversions than visitors
            conversions = st.number_input(f"Conversions", value=100 if i == 0 else 110, min_value=0, max_value=max_conv, key=f"conversions_bin_{i}")
        
        binary_variants.append({
            "name": name,
            "visitors": visitors,
            "conversions": conversions
        })
        
        # Add a small divider between variants
        if i < num_variants - 1:
            st.markdown("<hr style='margin: 10px 0; opacity: 0.3;'>", unsafe_allow_html=True)

if metric_type == "Continuous (Revenue/Value)":
    with tabs[1]:  # Continuous metrics tab
        st.markdown("##### 📊 Enter visitor and value data for each variant")
        
        continuous_variants = []
        
        # Create a more compact form layout
        for i in range(num_variants):
            variant_name = "Control" if i == 0 else f"Variant {i}"
            
            cols = st.columns([1.5, 1, 1, 1])
            with cols[0]:
                name = st.text_input(f"Variant name", value=variant_name, key=f"name_cont_{i}")
            with cols[1]:
                visitors = st.number_input(f"Visitors", value=1000 if i == 0 else 1000, min_value=1, key=f"visitors_cont_{i}")
            with cols[2]:
                mean_value = st.number_input(f"Mean value", value=10.0 if i == 0 else 11.0, min_value=0.0, step=0.1, key=f"mean_{i}")
            with cols[3]:
                std_dev = st.number_input(f"Std dev", value=5.0, min_value=0.01, step=0.1, key=f"std_dev_{i}")
            
            continuous_variants.append({
                "name": name,
                "visitors": visitors,
                "mean": mean_value,
                "std_dev": std_dev
            })
            
            # Add a small divider between variants
            if i < num_variants - 1:
                st.markdown("<hr style='margin: 10px 0; opacity: 0.3;'>", unsafe_allow_html=True)

# Calculate button
st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("Calculate Significance", type="primary", use_container_width=True)

# Default alpha level for statistical significance
alpha = 0.05
one_tailed = (hypothesis_type == "One-tailed")

if analyze_btn:
    st.markdown('<div class="sub-header">Test Results</div>', unsafe_allow_html=True)
    
    # Perform analysis based on metric type
    if metric_type == "Binary (Conversion)":
        results = analyze_binary_test(binary_variants, 
                                     alpha=alpha, 
                                     one_tailed=one_tailed, 
                                     correction=apply_correction)
        
        # Check for low sample size warning
        low_sample = False
        for variant in results:
            if variant["visitors"] < min_sample_warning:
                low_sample = True
        
        if low_sample:
            st.warning(f"⚠️ Low sample size detected for one or more variants. Results may not be reliable.")
        
        # Create summary of results in natural language
        variant_results = [r for r in results if not r["is_control"]]
        if variant_results:
            control_name = results[0]["name"]
            for result in variant_results:
                summary = generate_result_summary(result, control_name, "convert")
                
                # Display the summary
                st.markdown(f"""
                <div class="results-summary">
                    {summary}
                </div>
                """, unsafe_allow_html=True)
                
                # Create a simple table for key metrics
                metrics_df = pd.DataFrame([
                    {"Metric": "Visitors", control_name: results[0]["visitors"], result["name"]: result["visitors"]},
                    {"Metric": "Conversions", control_name: results[0]["conversions"], result["name"]: result["conversions"]},
                    {"Metric": "Conversion Rate", 
                     control_name: f"{results[0]['conversion_rate']*100:.2f}%", 
                     result["name"]: f"{result['conversion_rate']*100:.2f}%"},
                    {"Metric": "Relative Uplift", control_name: "baseline", result["name"]: f"{result['relative_effect']:+.2f}%"},
                    {"Metric": "Confidence Level", control_name: "N/A", result["name"]: f"{result['confidence_level']:.1f}%"},
                    {"Metric": "Statistical Power", control_name: "N/A", result["name"]: f"{result['power']:.1f}%"}
                ])
                
                st.dataframe(metrics_df.set_index("Metric"), use_container_width=True)
        
    else:  # Continuous metric analysis
        results = analyze_continuous_test(continuous_variants, 
                                        alpha=alpha, 
                                        one_tailed=one_tailed, 
                                        correction=apply_correction)
        
        # Check for low sample size warning
        low_sample = False
        for variant in results:
            if variant["visitors"] < min_sample_warning:
                low_sample = True
        
        if low_sample:
            st.warning(f"⚠️ Low sample size detected for one or more variants. Results may not be reliable.")
        
        # Create summary of results in natural language
        variant_results = [r for r in results if not r["is_control"]]
        if variant_results:
            control_name = results[0]["name"]
            for result in variant_results:
                summary = generate_result_summary(result, control_name, "perform")
                
                # Display the summary
                st.markdown(f"""
                <div class="results-summary">
                    {summary}
                </div>
                """, unsafe_allow_html=True)
                
                # Create a simple table for key metrics
                metrics_df = pd.DataFrame([
                    {"Metric": "Visitors", control_name: results[0]["visitors"], result["name"]: result["visitors"]},
                    {"Metric": "Mean Value", 
                     control_name: f"{results[0]['mean']:.2f}", 
                     result["name"]: f"{result['mean']:.2f}"},
                    {"Metric": "Standard Deviation", 
                     control_name: f"{results[0]['std_dev']:.2f}", 
                     result["name"]: f"{result['std_dev']:.2f}"},
                    {"Metric": "Relative Uplift", control_name: "baseline", result["name"]: f"{result['relative_effect']:+.2f}%"},
                    {"Metric": "Confidence Level", control_name: "N/A", result["name"]: f"{result['confidence_level']:.1f}%"},
                    {"Metric": "Statistical Power", control_name: "N/A", result["name"]: f"{result['power']:.1f}%"}
                ])
                
                st.dataframe(metrics_df.set_index("Metric"), use_container_width=True)
        
    st.markdown("""
    ### How to interpret test reliability
    
    - **Below 50%**: Sample size is too small, results are unreliable
    - **50-80%**: Moderate reliability, but more data would be beneficial
    - **Above 80%**: Good reliability, sufficient sample size to detect the observed effect
    
    Higher test reliability means you can be more confident that the observed uplift is real and not due to random chance.
    """)
