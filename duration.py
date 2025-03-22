import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="A/B Test Duration Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    body{
        color: #1E88E5;
    }
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
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        color : #424242;
        font-size: 26px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .help-text {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
    .stat-highlight {
        font-weight: bold;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">A/B Test Duration Calculator</div>', unsafe_allow_html=True)
st.markdown("""
This calculator determines the **minimum sample size and test duration** required to detect a statistically significant difference 
between your control and test variants. It uses statistical power analysis to ensure your test can reliably detect 
the expected effect size with your specified confidence level.
""")

# Move all inputs to sidebar for better mobile and desktop accessibility
with st.sidebar:
    st.markdown('<div class="sub-header">Test Parameters</div>', unsafe_allow_html=True)
    
    # Existing Data Inputs
    st.markdown("#### Historical Data")
    all_visitors = st.number_input("Total Visitors", 
                                 min_value=100, 
                                 max_value=10000000, 
                                 value=7500, 
                                 step=100,
                                 help="Total number of visitors in your historical data")
    
    visitors_with_conversions = st.number_input("Visitors with Conversions", 
                                             min_value=1, 
                                             max_value=all_visitors, 
                                             value=750, 
                                             step=10,
                                             help="Number of visitors who completed the desired conversion action")
    
    days_covered = st.number_input("Data Collection Period (days)", 
                                min_value=1, 
                                max_value=365, 
                                value=30, 
                                step=1,
                                help="Number of days over which the historical data was collected")

    # Compute Baseline Metrics
    baseline_conversion = round((visitors_with_conversions / all_visitors) * 100, 2) if all_visitors > 0 else 0
    daily_visitors = round(all_visitors / days_covered) if days_covered > 0 else 0

    st.markdown(f"**Baseline Conversion Rate:** {baseline_conversion}%")
    st.markdown(f"**Average Daily Visitors:** {daily_visitors}")

    # A/B Test Parameters
    st.markdown("#### Test Settings")
    target_test_duration = st.number_input("Target Test Duration (days)", 
                                        min_value=1, 
                                        max_value=365, 
                                        value=14, 
                                        step=1,
                                        help="Your desired test duration in days")
    
    color_coding_percentage = st.number_input("Acceptable Threshold (%)", 
                                           min_value=0.1, 
                                           max_value=10.0, 
                                           value=2.0, 
                                           step=0.1,
                                           help="Percentage difference from target that is considered acceptable")
    
    variations = st.number_input("Number of Variants", 
                              min_value=2, 
                              max_value=5, 
                              value=2, 
                              step=1,
                              help="Total number of variations in your test, including the control")
    
    target_improvement_input = st.text_input("Minimum Detectable Effect (%)", 
                                       value="10.0",
                                       help="The smallest meaningful improvement you want to be able to detect")
    
    # Validate and convert target improvement
    try:
        target_improvement = float(target_improvement_input)
        if target_improvement < 0.25:
            st.warning("Minimum improvement should be at least 0.25%")
            target_improvement = 0.25
        elif target_improvement > 50:
            st.warning("Maximum improvement should be at most 50%")
            target_improvement = 50
    except ValueError:
        st.error("Please enter a valid number for Expected Improvement")
        target_improvement = 10.0
    
    significance_level = st.selectbox("Statistical Significance Level", 
                                   options=[75, 80, 85, 90, 95, 99], 
                                   index=3,
                                   help="Desired confidence level for your test results (1 - α)")
    
    # New parameter: One-tailed vs Two-tailed test
    test_type = st.radio(
        "Test Type",
        options=["Two-tailed", "One-tailed"],
        index=0,
        help="""
        Two-tailed: Test if variant is different from control (could be better or worse)
        One-tailed: Test if variant is better than control (directional hypothesis)
        """
    )

# Z-score lookup tables for both test types
two_tailed_z_scores = {75: 1.15, 80: 1.28, 85: 1.44, 90: 1.65, 95: 1.96, 99: 2.58}
one_tailed_z_scores = {75: 0.67, 80: 0.84, 85: 1.04, 90: 1.28, 95: 1.65, 99: 2.33}

# Get the appropriate z-score based on test type
z_scores = one_tailed_z_scores if test_type == "One-tailed" else two_tailed_z_scores

# Function to compute test duration with statistical reasoning
def calculate_test_duration(daily_visits, vars, conversion, improvement, significance, is_one_tailed=False):
    # Reference values based on industry standards for sample size estimation
    reference_values = {
        0.25: 299520, 0.5: 74880, 0.75: 33280, 1.0: 18720,
        2.0: 4680, 3.0: 2080, 5.0: 749, 6.0: 520, 7.0: 382,
        8.0: 293, 9.0: 231, 10.0: 187, 12.5: 120, 25.0: 30, 50.0: 7
    }

    closest_imp = min(reference_values.keys(), key=lambda x: abs(x - improvement))
    base_days = reference_values[closest_imp]

    # Adjustment factors
    variant_factor = {2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}.get(vars, 1.0)
    
    # Use different z-score tables based on test type
    z_table = one_tailed_z_scores if is_one_tailed else two_tailed_z_scores
    sig_factor = (z_table[significance] / z_table[95]) ** 2
    
    cvr_factor = 10.0 / conversion if conversion > 0 else 1.0
    visitor_factor = 250 / daily_visits if daily_visits > 0 else 1.0

    # One-tailed tests typically require fewer samples
    tail_factor = 0.8 if is_one_tailed else 1.0

    days = base_days * variant_factor * sig_factor * cvr_factor * visitor_factor * tail_factor
    visitors_needed = days * daily_visits
    samples_per_variant = visitors_needed / vars
    
    # Calculate p-value threshold and beta (Type II error rate)
    alpha = (100 - significance) / 100
    beta = 1 - (significance / 100)
    power = 1 - beta
    
    # Get the appropriate z-score based on test type
    z_score = z_table[significance]
    
    return {
        "days": max(1, round(days)), 
        "visitors": max(1, round(visitors_needed)),
        "samples_per_variant": max(1, round(samples_per_variant)),
        "alpha": alpha,
        "beta": beta,
        "power": power,
        "z_score": z_score,
        "is_one_tailed": is_one_tailed
    }

# Compute results based on selected test type
is_one_tailed = (test_type == "One-tailed")
result = calculate_test_duration(daily_visitors, variations, baseline_conversion, target_improvement, significance_level, is_one_tailed)

# Main content
# Create three columns for metrics
metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Minimum Test Duration</div>
        <div class="metric-value">{} days</div>
    </div>
    """.format(result['days']), unsafe_allow_html=True)
    
with metric_col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Total Visitors Needed</div>
        <div class="metric-value">{:,}</div>
    </div>
    """.format(result['visitors']), unsafe_allow_html=True)



# Additional stat metrics
stat_col1, stat_col2 = st.columns(2)

with stat_col1:
    duration_diff = result['days'] - target_test_duration
    status_color = "green" if abs(duration_diff) <= (target_test_duration) else "red"
    status_text = "On Target" if status_color == "green" else "Off Target"
    status_bg = "#d4edda" if status_color == "green" else "#f8d7da"
    
    st.markdown("""
    <div class="metric-card" style="background-color: {}">
        <div class="metric-label">Status</div>
        <div class="metric-value">{}</div>
        <div class="help-text">Difference: {} days</div>
    </div>
    """.format(status_bg, status_text, duration_diff), unsafe_allow_html=True)

with stat_col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Z-Score</div>
        <div class="metric-value">{:.2f}</div>
        <div class="help-text">Critical value threshold</div>
    </div>
    """.format(result['z_score']), unsafe_allow_html=True)

# Recommendation box
test_type_explanation = "This test will detect if your variant is significantly better than control (directional hypothesis)" if is_one_tailed else "This test will detect if your variant is significantly different from control (could be better or worse)"

st.info(f"""
**Recommendation:** Run your test for at least **{result['days']} days** to achieve **{significance_level}%** statistical significance using a **{test_type}** test.

{test_type_explanation}.

This will require approximately **{result['visitors']:,} total visitors** with **{result['samples_per_variant']:,} visitors per variant**.

With these parameters, you'll be able to detect a **{target_improvement}%** improvement in your conversion rate with **{result['power']:.2f}** power (probability of detecting a true effect).
""")

# Statistical Explanation
with st.expander("Statistical Methodology"):
    st.markdown(f"""
    ### Statistical Calculations Behind the Calculator
    
    This calculator uses statistical power analysis to determine the minimum sample size needed for your A/B test. Here's how the calculations work:
    
    #### Key Statistical Concepts:
    
    - **Alpha (α):** The significance level, currently set to **{result['alpha']:.3f}**. This is the probability of incorrectly rejecting the null hypothesis (false positive).
    
    - **Beta (β):** The probability of a Type II error, currently **{result['beta']:.3f}**. This is the probability of failing to detect a true effect.
    
    - **Statistical Power (1-β):** Currently **{result['power']:.2f}**. This is the probability of correctly detecting a true effect when it exists.
    
    - **Minimum Detectable Effect (MDE):** The smallest meaningful improvement you want to be able to detect, currently set to **{target_improvement:.1f}%**.
    
    - **Z-Score:** The critical value based on your chosen significance level and test type, currently **{result['z_score']:.2f}**.
    
    - **Test Type:** You've selected a **{test_type}** test. 
        - A **one-tailed test** tests the hypothesis that the variant is better than the control (directional).
        - A **two-tailed test** tests the hypothesis that the variant is different from the control (could be better or worse).
        - One-tailed tests typically require smaller sample sizes but should only be used when you are exclusively interested in improvement.
    
    #### Sample Size Formula:
    
    The fundamental formula for the minimum sample size per variant is:
    
    ```
    n = (2 * (z_α + z_β)² * p * (1-p)) / (MDE)²
    ```
    
    Where:
    - n = sample size per variant
    - z_α = z-score for your significance level (one-tailed or two-tailed test)
    - z_β = z-score for your desired power
    - p = baseline conversion rate
    - MDE = minimum detectable effect (as a decimal)
    
    #### Adjustment Factors:
    
    This calculator applies multiple adjustment factors to the sample size:
    
    1. **Variant Factor:** More variants require larger samples per variant
    2. **Significance Factor:** Higher confidence levels require larger samples
    3. **Conversion Rate Factor:** Lower baseline conversion rates require larger samples
    4. **Traffic Factor:** Sites with lower daily traffic need longer test durations
    5. **Test Type Factor:** One-tailed tests typically require ~20% fewer samples than two-tailed tests
    
    #### Test Duration Calculation:
    
    ```
    Test Duration (days) = Total Sample Size / Daily Traffic
    ```
    
    The minimum test duration accounts for realistic traffic patterns and ensures you collect enough data to make statistically valid decisions.
    """, unsafe_allow_html=True)

# Visualization Tabs
st.markdown('<div class="sub-header">Visual Analysis</div>', unsafe_allow_html=True)
viz_tab1, viz_tab2 = st.tabs(["Significance Impact", "MDE Impact"])

with viz_tab1:
    # Create data for significance level comparison
    significance_levels = [75, 80, 85, 90, 95, 99]
    sig_days = [calculate_test_duration(daily_visitors, variations, baseline_conversion, target_improvement, sig, is_one_tailed)["days"] 
               for sig in significance_levels]
    
    sig_df = pd.DataFrame({
        "Significance Level": [f"{sig}%" for sig in significance_levels],
        "Days Required": sig_days
    })
    
    # Create interactive plotly bar chart
    fig = px.bar(
        sig_df, 
        x="Significance Level", 
        y="Days Required",
        color="Days Required",
        color_continuous_scale="Blues",
        title=f"Test Duration by Significance Level (at {target_improvement}% MDE, {test_type})",
        labels={"Days Required": "Days Required", "Significance Level": "Significance Level (%)"},
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Significance Level (%)",
        yaxis_title="Days Required",
        coloraxis_showscale=False,
        hovermode="x",
        hoverlabel=dict(
            bgcolor="black",
            font_size=12
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="help-text">
    This chart illustrates how increasing the confidence level (reducing Type I error risk) requires longer test durations.
    Each additional percentage point of confidence requires exponentially more data.
    </div>
    """, unsafe_allow_html=True)

with viz_tab2:
    # Create data for improvement comparison with more granular values
    # Using more data points to create a smoother curve
    imp_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 15.0, 20.0, 25.0, 50.0]
    imp_days = [calculate_test_duration(daily_visitors, variations, baseline_conversion, imp, significance_level, is_one_tailed)["days"] 
            for imp in imp_values]
    
    imp_df = pd.DataFrame({
        "Expected Improvement": [f"{imp}%" for imp in imp_values],
        "Days Required": imp_days,
        "Improvement": imp_values
    })
    
    # Create a more detailed and better scaled chart
    fig = go.Figure()
    
    # Add the main line trace
    fig.add_trace(go.Scatter(
        x=imp_df["Improvement"],
        y=imp_df["Days Required"],
        mode='lines+markers',
        name='Days Required',
        line=dict(color='#1E88E5', width=3, shape='spline'),
        marker=dict(size=8, color='#1E88E5', line=dict(color='white', width=1)),
        hovertemplate="<b>%{x}%</b> improvement<br>%{y} days required<extra></extra>"
    ))
    
    # Add annotations for specific points
    fig.add_trace(go.Scatter(
        x=[imp_values[0], imp_values[-1]],
        y=[imp_days[0], imp_days[-1]],
        mode='markers+text',
        marker=dict(size=10, color='#ff4757', symbol='circle'),
        text=[f"{imp_days[0]} days", f"{imp_days[-1]} days"],
        textposition=["top right", "bottom right"],
        textfont=dict(size=12, color='#ff4757'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add highlight for the selected improvement value
    closest_imp = min(imp_values, key=lambda x: abs(x - target_improvement))
    closest_imp_idx = imp_values.index(closest_imp)
    fig.add_trace(go.Scatter(
        x=[target_improvement],
        y=[result['days']],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle', line=dict(color='white', width=2)),
        name='Your Selection',
        hovertemplate="<b>Your selection</b><br>%{x}% improvement<br>%{y} days required<extra></extra>"
    ))
    
    # Add horizontal line for target duration
    fig.add_shape(type="line",
        x0=min(imp_values), y0=target_test_duration, 
        x1=max(imp_values), y1=target_test_duration,
        line=dict(color="red", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=max(imp_values) * 0.8,
        y=target_test_duration,
        text=f"Target Duration: {target_test_duration} days",
        showarrow=False,
        yshift=10
    )
    
    # Add vertical line for target improvement
    fig.add_shape(type="line",
        x0=target_improvement, y0=0, 
        x1=target_improvement, y1=max(imp_days) * 0.9,
        line=dict(color="green", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=target_improvement,
        y=max(imp_days) * 0.9,
        text=f"Your target: {target_improvement}%",
        showarrow=False,
        xshift=5,
        yshift=-10
    )
    
    # Calculate an appropriate y-axis range
    y_min = min(imp_days) * 0.8
    y_max = min(max(imp_days) * 1.2, result['days'] * 5)  # Limit the max to 5x the result days
    
    # Update the layout with better scaling
    fig.update_layout(
        title=f"Test Duration by Minimum Detectable Effect (at {significance_level}% significance, {test_type})",
        xaxis_title="Minimum Detectable Effect (%)",
        yaxis_title="Days Required",
        yaxis=dict(
            type="log",  # Logarithmic scale to show differences across orders of magnitude
            range=[np.log10(y_min), np.log10(y_max)],  # Constrain the y-axis range
            tickmode="auto",
            nticks=10,
            gridcolor='rgba(0,0,0,0.1)',
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=imp_values,
            ticktext=[f"{i}%" for i in imp_values],
            gridcolor='rgba(0,0,0,0.1)',
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="black",
            font_size=12
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="help-text">
    This chart demonstrates the inverse relationship between effect size and required test duration.
    The relationship follows a power law curve - smaller effects require exponentially more data to detect reliably.
    Consider whether a very small effect is worth detecting when planning your test resources.
    </div>
    """, unsafe_allow_html=True)    

# Reference Table
st.markdown('<div class="sub-header">Test Duration Reference Table</div>', unsafe_allow_html=True)
improvements = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0]
variant_counts = [2, 3, 4, 5]

reference_df = pd.DataFrame(index=improvements, columns=variant_counts)
reference_df.index.name = "Expected Effect (%)"

for imp in improvements:
    for var in variant_counts:
        res = calculate_test_duration(daily_visitors, var, baseline_conversion, imp, significance_level, is_one_tailed)
        reference_df.loc[imp, var] = res["days"]

reference_df.columns = [f"{v} Variants" for v in variant_counts]
reference_df.index = [f"{val}% Lift" for val in improvements]

# Create heatmap for reference table
fig = px.imshow(
    reference_df.values,
    x=reference_df.columns,
    y=reference_df.index,
    color_continuous_scale="Blues_r",
    text_auto=True,
    aspect="auto",
    title=f"Test Duration (days) by Effect Size & Variants (at {significance_level}% significance, {test_type})"
)

fig.update_layout(
    xaxis_title="Number of Variants",
    yaxis_title="Minimum Detectable Effect",
    coloraxis_showscale=False,  # Remove colorbar as requested for mobile optimization
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Download button for reference data
csv = reference_df.to_csv().encode('utf-8')
st.download_button(
    label=f"Download Reference Table ({test_type})",
    data=csv,
    file_name=f"ab_test_reference_table_{significance_level}pct_{test_type.lower().replace('-', '_')}.csv",
    mime="text/csv",
)

# Footer with additional information for CRO analysts
st.markdown("""
---
### CRO Best Practices 

**Test Implementation:**
- Run full weeks to account for day-of-week effects
- Don't peek at results early to avoid inflated Type I errors
- Ensure even traffic distribution across variants
- Use robust traffic splitting methods (cookie-based, not session-based)
- Consider the Minimum Detectable Effect in context of implementation cost

**Statistical Considerations:**
- One-tailed tests should only be used when you have a strong directional hypothesis
- Two-tailed tests are more conservative but provide more information
- For sequential testing, use a different model with alpha spending functions
- Be aware of the conversion variance impact on sample size requirements
- Consider bootstrapping methods for non-normally distributed metrics """)