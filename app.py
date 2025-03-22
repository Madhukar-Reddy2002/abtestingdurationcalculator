import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
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
    .result-positive {
        color: #28a745;
        font-weight: bold;
    }
    .result-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .result-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">A/B Test Significance Calculator</div>', unsafe_allow_html=True)
st.markdown("""
This calculator determines the **statistical significance** of your A/B test results. 
It analyzes your test data to determine if the observed differences between your control and variant 
are statistically significant or likely due to random chance.
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
    
    significance_level = st.select_slider(
        "Confidence Level",
        options=[90, 95, 99],
        value=95,
        help="Desired confidence level for your test results"
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
                "confidence_interval": (0, 0)
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
            "confidence_interval": (ci_lower, ci_upper)
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
                "confidence_interval": (0, 0)
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
            "confidence_interval": (ci_lower, ci_upper)
        }
        results.append(result)
    
    return results

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

# Convert significance level to alpha
alpha = (100 - significance_level) / 100
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
        
        # Create a dashboard-style layout
        # First row: Key metrics
        col1, col2, col3 = st.columns(3)
        
        control_rate = results[0]["conversion_rate"] * 100
        confidence_text = f"{significance_level}%"
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Control Conversion Rate</div>
                <div class="metric-value">{control_rate:.2f}%</div>
                <div class="metric-label">{results[0]['conversions']} / {results[0]['visitors']} visitors</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Find the best performing variant
            best_variant = None
            best_lift = 0
            
            for result in results[1:]:
                if result["relative_effect"] > best_lift:
                    best_lift = result["relative_effect"]
                    best_variant = result
            
            if best_variant:
                best_rate = best_variant["conversion_rate"] * 100
                status_class = "result-positive" if best_variant["significant"] else "result-neutral"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Best Variant: {best_variant['name']}</div>
                    <div class="metric-value">{best_rate:.2f}%</div>
                    <div class="metric-label {status_class}">Lift: {best_lift:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            correction_text = "with Bonferroni correction" if apply_correction and num_variants > 2 else ""
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Test Configuration</div>
                <div class="metric-value">{confidence_text} Confidence</div>
                <div class="metric-label">{hypothesis_type} test {correction_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a comprehensive comparison table
        st.markdown("### 📋 Detailed Comparison")
        
        table_data = []
        for result in results:
            variant_rate = result["conversion_rate"] * 100
            
            if result["is_control"]:
                status = "Control"
                status_class = ""
                rel_lift = ""
                ci_text = ""
                p_value = ""
            else:
                lift = result["relative_effect"]
                p_value = result["p_value"]
                
                if result["significant"]:
                    if lift > 0:
                        status = "✅ Winner"
                        status_class = "result-positive"
                    else:
                        status = "❌ Loser"
                        status_class = "result-negative"
                else:
                    status = "⚖️ Inconclusive"
                    status_class = "result-neutral"
                
                rel_lift = f"{lift:+.2f}%"
                ci_lower, ci_upper = result["confidence_interval"]
                ci_lower_pct = ci_lower * 100
                ci_upper_pct = ci_upper * 100
                ci_text = f"[{ci_lower_pct:.2f}%, {ci_upper_pct:.2f}%]"
                p_value = f"{p_value:.4f}"
            
            table_data.append({
                "Variant": result["name"],
                "Status": status,
                "Status Class": status_class,
                "Visitors": result["visitors"],
                "Conversions": result["conversions"],
                "Rate": f"{variant_rate:.2f}%",
                "Relative Lift": rel_lift,
                "Confidence Interval": ci_text,
                "p-value": p_value
            })
        
        # Convert to DataFrame for display
        df = pd.DataFrame(table_data)
        
        # Create styled DataFrame
        def highlight_status(val):
            if val == "✅ Winner":
                return 'background-color: #d4edda; color: #155724'
            elif val == "❌ Loser":
                return 'background-color: #f8d7da; color: #721c24'
            elif val == "⚖️ Inconclusive":
                return 'background-color: #e2e3e5; color: #383d41'
            elif val == "Control":
                return 'background-color: #cce5ff; color: #004085'
            return ''
            
        # Display the styled table
        styled_df = df[["Variant", "Status", "Visitors", "Conversions", "Rate", "Relative Lift", "Confidence Interval", "p-value"]]
        st.dataframe(styled_df.style.applymap(highlight_status, subset=['Status']), use_container_width=True)
        
        # Create visualization
        st.markdown("### 📊 Conversion Rate Visualization")
        
        # Prepare data for visualization
        viz_data = []
        for result in results:
            variant_status = "Control" if result["is_control"] else (
                             "Winner" if result["significant"] and result["effect"] > 0 else (
                             "Loser" if result["significant"] and result["effect"] < 0 else "Inconclusive"))
            
            viz_data.append({
                "Variant": result["name"],
                "Conversion Rate (%)": result["conversion_rate"] * 100,
                "Status": variant_status,
                "Visitors": result["visitors"],
                "Conversions": result["conversions"]
            })
        
        viz_df = pd.DataFrame(viz_data)
        
        # Create color map
        color_map = {
            "Control": "#1E88E5",  # Blue
            "Winner": "#28a745",   # Green
            "Loser": "#dc3545",    # Red
            "Inconclusive": "#6c757d"  # Gray
        }
        
        # Create bar chart with confidence intervals
        fig = go.Figure()
        
        # Add bars
        for _, row in viz_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Variant"]],
                y=[row["Conversion Rate (%)"]],
                name=row["Variant"],
                marker_color=color_map[row["Status"]],
                text=[f"{row['Conversion Rate (%)']:.2f}%"],
                textposition="auto",
                hovertemplate=f"<b>{row['Variant']} ({row['Status']})</b><br>Conversion Rate: {row['Conversion Rate (%)']:.2f}%<br>Visitors: {row['Visitors']}<br>Conversions: {row['Conversions']}<extra></extra>"
            ))
        
        # Add error bars for confidence intervals if not control
        for i, result in enumerate(results):
            if not result["is_control"]:
                ci_lower, ci_upper = result["confidence_interval"]
                ci_lower_pct = (result["conversion_rate"] + ci_lower) * 100
                ci_upper_pct = (result["conversion_rate"] + ci_upper) * 100
                
                # Add line for confidence interval
                fig.add_trace(go.Scatter(
                    x=[result["name"], result["name"]],
                    y=[ci_lower_pct, ci_upper_pct],
                    mode="lines",
                    marker=dict(color="black"),
                    line=dict(width=2),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                
                # Add markers for confidence interval boundaries
                for y_val in [ci_lower_pct, ci_upper_pct]:
                    fig.add_trace(go.Scatter(
                        x=[result["name"]],
                        y=[y_val],
                        mode="markers",
                        marker=dict(color="black", size=6),
                        showlegend=False,
                        hovertemplate=f"CI Boundary: {y_val:.2f}%<extra></extra>"
                    ))
        
        # Add a horizontal line for the control rate
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=control_rate,
            x1=len(results) - 0.5,
            y1=control_rate,
            line=dict(color="#1E88E5", width=2, dash="dash"),
        )
        
        # Update layout
        fig.update_layout(
            title="Conversion Rate by Variant with Confidence Intervals",
            xaxis_title="Variant",
            yaxis_title="Conversion Rate (%)",
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(0,0,0,0.2)",
            ),
            hovermode="closest",
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False,
            plot_bgcolor="white",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical interpretation explanation
        with st.expander("📘 How to interpret these results"):
            st.markdown("""
            ### Understanding A/B Test Results
            
            #### Statistical Significance
            - **p-value < {alpha}**: We can be {significance_level}% confident that the observed difference is not due to random chance
            - **Confidence Interval**: The range within which the true difference is likely to fall with {significance_level}% confidence
            
            #### What the results mean
            - **Winner (✅)**: The variant outperformed the control with statistical significance
            - **Loser (❌)**: The variant underperformed compared to the control with statistical significance
            - **Inconclusive (⚖️)**: We cannot determine with confidence whether the variant is better or worse than the control
            
            #### Next steps
            - **For winners**: Consider implementing the winning variant
            - **For losers**: Avoid implementing or revise the approach
            - **For inconclusive results**: Consider running the test longer to collect more data
            """.format(alpha=alpha, significance_level=significance_level))
        
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
        
        # Create a dashboard-style layout
        # First row: Key metrics
        col1, col2, col3 = st.columns(3)
        
        control_mean = results[0]["mean"]
        confidence_text = f"{significance_level}%"
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Control Mean Value</div>
                <div class="metric-value">{control_mean:.2f}</div>
                <div class="metric-label">{results[0]['visitors']} visitors</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Find the best performing variant
            best_variant = None
            best_lift = 0
            
            for result in results[1:]:
                if result["relative_effect"] > best_lift:
                    best_lift = result["relative_effect"]
                    best_variant = result
            
            if best_variant:
                best_mean = best_variant["mean"]
                status_class = "result-positive" if best_variant["significant"] else "result-neutral"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Best Variant: {best_variant['name']}</div>
                    <div class="metric-value">{best_mean:.2f}</div>
                    <div class="metric-label {status_class}">Lift: {best_lift:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            correction_text = "with Bonferroni correction" if apply_correction and num_variants > 2 else ""
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Test Configuration</div>
                <div class="metric-value">{confidence_text} Confidence</div>
                <div class="metric-label">{hypothesis_type} test {correction_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a comprehensive comparison table
        st.markdown("### 📋 Detailed Comparison")
        
        table_data = []
        for result in results:
            if result["is_control"]:
                status = "Control"
                status_class = ""
                rel_lift = ""
                ci_text = ""
                p_value = ""
            else:
                lift = result["relative_effect"]
                p_value = result["p_value"]
                
                if result["significant"]:
                    if lift > 0:
                        status = "✅ Winner"
                        status_class = "result-positive"
                    else:
                        status = "❌ Loser"
                        status_class = "result-negative"
                else:
                    status = "⚖️ Inconclusive"
                    status_class = "result-neutral"
                
                rel_lift = f"{lift:+.2f}%"
                ci_lower, ci_upper = result["confidence_interval"]
                ci_text = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
                p_value = f"{p_value:.4f}"
            
            table_data.append({
                "Variant": result["name"],
                "Status": status,
                "Status Class": status_class,
                "Visitors": result["visitors"],
                "Mean Value": f"{result['mean']:.2f}",
                "Std Dev": f"{result.get('std_dev', 0):.2f}",
                "Relative Lift": rel_lift,
                "Confidence Interval": ci_text,
                "p-value": p_value
            })
        
        # Convert to DataFrame for display
        df = pd.DataFrame(table_data)
        
        # Create styled DataFrame
        def highlight_status(val):
            if val == "✅ Winner":
                return 'background-color: #d4edda; color: #155724'
            elif val == "❌ Loser":
                return 'background-color: #f8d7da; color: #721c24'
            elif val == "⚖️ Inconclusive":
                return 'background-color: #e2e3e5; color: #383d41'
            elif val == "Control":
                return 'background-color: #cce5ff; color: #004085'
            return ''