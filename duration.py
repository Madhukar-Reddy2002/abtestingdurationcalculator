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
    daily_visitors_override = st.number_input("Expected Daily Visitors During Test", 
                                          min_value=10, 
                                          max_value=1000000, 
                                          value=daily_visitors, 
                                          step=10,
                                          help="Expected daily visitors during the test period. Default is based on historical data.")
    
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
        if target_improvement < 0.1:
            st.warning("Minimum improvement should be at least 0.1%")
            target_improvement = 0.1
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
    
    # Test type
    test_type = st.radio(
        "Test Type",
        options=["Two-tailed", "One-tailed"],
        index=0,
        help="""
        Two-tailed: Test if variant is different from control (could be better or worse)
        One-tailed: Test if variant is better than control (directional hypothesis)
        """
    )
    
    # Add statistical power option
    statistical_power = st.selectbox("Statistical Power", 
                                  options=[70, 75, 80, 85, 90, 95], 
                                  index=2,
                                  help="Probability of detecting a true effect when it exists (1 - β)")

# Z-score lookup tables for both test types
two_tailed_z_scores = {75: 1.15, 80: 1.28, 85: 1.44, 90: 1.65, 95: 1.96, 99: 2.58}
one_tailed_z_scores = {75: 0.67, 80: 0.84, 85: 1.04, 90: 1.28, 95: 1.65, 99: 2.33}

# Power z-scores
power_z_scores = {70: 0.524, 75: 0.674, 80: 0.84, 85: 1.04, 90: 1.28, 95: 1.645}

# Function to calculate test duration with the updated formula
def calculate_test_duration(daily_visits, vars, baseline_cvr, improvement, significance, power, is_one_tailed=False):
    # Get z-scores based on test type and significance level
    z_table = one_tailed_z_scores if is_one_tailed else two_tailed_z_scores
    z_alpha = z_table[significance]
    z_beta = power_z_scores[power]
    
    # Convert percentage values to proportions
    p1 = baseline_cvr / 100.0  # Control conversion rate
    delta = improvement / 100.0  # Minimum detectable effect as proportion
    p2 = p1 * (1 + delta)  # Expected variant conversion rate
    
    # Calculate sample size per variant using the new formula
    # n = (Z1−α/2+Z1−β)²⋅[p1(1−p1)+p2(1−p2)]/(p2-p1)²
    numerator = 2* (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p2 - p1) ** 2
    
    # Avoid division by zero
    if denominator == 0:
        sample_size_per_variant = float('inf')
    else:
        sample_size_per_variant = numerator / denominator
    
    # Total sample size across all variants
    total_sample_size = sample_size_per_variant * vars
    
    # Calculate test duration in days
    days = total_sample_size / daily_visits if daily_visits > 0 else float('inf')
    
    # Calculate alpha, beta, and power
    alpha = (100 - significance) / 100
    beta = (100 - power) / 100
    power_value = 1 - beta
    
    return {
        "days": max(1, round(days)), 
        "visitors": max(1, round(total_sample_size)),
        "samples_per_variant": max(1, round(sample_size_per_variant)),
        "alpha": alpha,
        "beta": beta,
        "power": power_value,
        "z_alpha": z_alpha,
        "z_beta": z_beta,
        "is_one_tailed": is_one_tailed,
        "p1": p1,
        "p2": p2,
        "delta": delta
    }

# Compute results based on selected test type
is_one_tailed = (test_type == "One-tailed")
result = calculate_test_duration(daily_visitors_override, variations, baseline_conversion, 
                             target_improvement, significance_level, statistical_power, is_one_tailed)

# Main content
# Create columns for metrics
metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    days = result['days']
    weeks = round(days / 7)
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Minimum Test Duration</div>
        <div class="metric-value">{} days ({} Weeks)</div>
    </div>
    """.format(days , weeks), unsafe_allow_html=True)
    
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
    status_color = "green" if abs(duration_diff) <= (target_test_duration * color_coding_percentage / 100) else "red"
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
        <div class="metric-label">Visitors per Variant</div>
        <div class="metric-value">{:,}</div>
        <div class="help-text">Required for statistical validity</div>
    </div>
    """.format(result['samples_per_variant']), unsafe_allow_html=True)

# Recommendation box
test_type_explanation = "This test will detect if your variant is significantly better than control (directional hypothesis)" if is_one_tailed else "This test will detect if your variant is significantly different from control (could be better or worse)"

st.info(f"""
📌 **Recommendation:**  
Run your A/B test for **at least {result['days']} days** to reach **{significance_level}% significance** and **{statistical_power}% power** using a **{test_type} test**.

🧪 You'll need about **{result['visitors']:,} total visitors**,  
with **{result['samples_per_variant']:,} per variant**.

🎯 This setup can detect a **{target_improvement}% improvement**  
in your conversion rate with **{result['power']:.2f} probability** of catching a real difference.

{test_type_explanation}
""")

# Statistical Explanation
with st.expander("Statistical Methodology"):
    st.markdown(f"""
    ### 📊 Behind the Calculations
    
    This calculator uses proven A/B testing math to estimate how long your test should run and how many visitors you need.

    #### 🔑 Key Concepts:
    - **Alpha (α): {result['alpha']:.3f}** → Risk of a **false positive** (detecting a change that isn’t real).
    - **Beta (β): {result['beta']:.3f}** → Risk of a **false negative** (missing a real change).
    - **Power: {result['power']:.2f}** → Chance of catching a **real difference** if it exists.
    - **MDE (Minimum Detectable Effect): {target_improvement:.1f}%** → Smallest change you care to detect.
    - **Test Type:** **{test_type}**  
        - *One-tailed:* Only care about improvement.  
        - *Two-tailed:* Care about any change (better or worse).

    #### 📐 Sample Size Formula:
    ```
    n = 2*(Zα + Zβ)² × [p1(1−p1) + p2(1−p2)] / (p2 - p1)²
    ```
    - **p1:** Current conversion rate → {result['p1']:.4f}  
    - **p2:** Expected improved rate → {result['p2']:.4f}  
    - **Δ (Difference):** {result['delta']:.4f}

    This formula works better for:
    - Small improvements (< 5%)
    - Very low or very high conversion rates

    #### ⏳ Duration Calculation:
    ```
    Duration (days) = Total Visitors Needed / Daily Visitors
    ```

    This tells you how long to run your test with your current traffic to get valid results.

    ---
    ⚡ **Why it matters:**  
    Making decisions with too little data increases the risk of bad outcomes. This tool helps you test smarter — not just faster.
    """, unsafe_allow_html=True)

# Visualization Tabs
st.markdown('<div class="sub-header">Visual Analysis</div>', unsafe_allow_html=True)
viz_tab1, viz_tab2 = st.tabs(["Significance Impact", "MDE Impact"])

with viz_tab1:
    # Create data for significance level comparison
    significance_levels = [75, 80, 85, 90, 95, 99]
    sig_days = [calculate_test_duration(daily_visitors_override, variations, baseline_conversion, 
                                    target_improvement, sig, statistical_power, is_one_tailed)["days"] 
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
    # Create data for improvement comparison with specific values
    imp_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    imp_days = [calculate_test_duration(daily_visitors_override, variations, baseline_conversion, 
                                    imp, significance_level, statistical_power, is_one_tailed)["days"] 
            for imp in imp_values]
    
    imp_df = pd.DataFrame({
        "Expected Improvement": [f"{imp}%" for imp in imp_values],
        "Days Required": imp_days,
        "Improvement": imp_values
    })
    
    # Create chart
    fig = go.Figure()
    
    # Add the main line trace
    fig.add_trace(go.Scatter(
        x=imp_df["Improvement"],
        y=imp_df["Days Required"],
        mode='lines+markers',
        line=dict(color='#1E88E5', width=3, shape='spline'),
        marker=dict(size=8, color='#1E88E5', line=dict(color='white', width=1)),
        hovertemplate="<b>%{x}%</b> improvement<br>%{y} days required<extra></extra>",
        showlegend=False
    ))
    
    # Add highlight for the selected improvement value
    fig.add_trace(go.Scatter(
        x=[target_improvement],
        y=[result['days']],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle', line=dict(color='white', width=2)),
        hovertemplate="<b>Your selection</b><br>%{x}% improvement<br>%{y} days required<extra></extra>",
        showlegend=False
    ))
    
    # Add horizontal line for target duration
    fig.add_shape(type="line",
        x0=min(imp_values), y0=target_test_duration, 
        x1=max(imp_values), y1=target_test_duration,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add vertical line for target improvement
    fig.add_shape(type="line",
        x0=target_improvement, y0=0, 
        x1=target_improvement, y1=max(imp_days) * 0.9,
        line=dict(color="green", width=2, dash="dash"),
    )
    
    # Calculate appropriate y-axis range
    y_min = min(imp_days) * 0.8
    y_max = min(max(imp_days) * 1.2, result['days'] * 5)
    
    # Update layout with custom ticks
    fig.update_layout(
        title="Test Duration by Minimum Detectable Effect",
        xaxis_title="Effect Size (%)",
        yaxis_title="Days",
        yaxis=dict(
            type="log",
            range=[np.log10(y_min), np.log10(y_max)],
            tickmode="auto",
            nticks=6,
            gridcolor='rgba(0,0,0,0.1)',
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            ticktext=["1%", "2%", "3%", "4%", "5%", "6%", "7%", "8%", "9%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%"],
            gridcolor='rgba(0,0,0,0.1)',
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="black",
            font_size=10
        ),
        height=350,
        margin=dict(l=30, r=30, t=50, b=50),
        autosize=True,
    )
    
    # Make the chart take up full width
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="help-text">
    Red line = target duration. Green line = your selected improvement.
    </div>
    """, unsafe_allow_html=True)    

# Reference Table
st.markdown('<div class="sub-header">Test Duration Reference Table</div>', unsafe_allow_html=True)
improvements = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0]
variant_counts = [2, 3, 4, 5]

reference_df = pd.DataFrame(index=improvements, columns=variant_counts)
reference_df.index.name = "Expected Effect (%)"

for imp in improvements:
    for var in variant_counts:
        res = calculate_test_duration(daily_visitors_override, var, baseline_conversion, 
                                  imp, significance_level, statistical_power, is_one_tailed)
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
    title=f"Test Duration (days) by Effect Size & Variants (at {significance_level}% significance, {statistical_power}% power, {test_type})"
)

fig.update_layout(
    xaxis_title="Number of Variants",
    yaxis_title="Minimum Detectable Effect",
    coloraxis_showscale=False,  # Remove colorbar for mobile optimization
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Download button for reference data
csv = reference_df.to_csv().encode('utf-8')
st.download_button(
    label=f"Download Reference Table ({test_type})",
    data=csv,
    file_name=f"ab_test_reference_table_{significance_level}pct_{statistical_power}pct_power_{test_type.lower().replace('-', '_')}.csv",
    mime="text/csv",
)

# Footer with additional information for CRO analysts
st.markdown("""
---
### 🧠 CRO Best Practices: What Smart Testers Always Do

#### ✅ Test Setup Tips:
- 📅 **Run full weeks** — capture weekday & weekend behavior.
- 🙈 **Don’t peek early!** It skews results and increases false positives.
- ⚖️ **Split traffic evenly** between variants.
- 🍪 **Use cookie-based** traffic splitting (more reliable than sessions).
- 💸 **Think beyond stats** — is the uplift worth the cost?

#### 📊 Smart Stats Reminders:
- 📈 We use a **more accurate formula** — great for small effect sizes or tricky conversion rates.
- 🎯 **One-tailed test?** Only if you're *sure* the variant will outperform.
- 🔁 **Two-tailed test?** Safer if you're testing for *any* difference.
- 🧪 For **ongoing checks (sequential tests)**, use specialized models like alpha-spending.
- 🐢 Low traffic? Better to run longer than to accept weak power.
- 👥 Testing a small segment? You'll need a **bigger impact** to move the overall needle.

---
""")