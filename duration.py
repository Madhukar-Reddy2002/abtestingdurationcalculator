import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Set page config
st.set_page_config(page_title="A/B Test Analyzer", layout="wide")

# Enhanced Custom CSS with better responsive design and more prominent headers
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Enhanced Title Styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1565C0;
    }
    
    /* Enhanced Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1976D2;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1976D2;
        padding-bottom: 0.5rem;
    }
    
    /* Enhanced Subsection Headers */
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2196F3;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Redesigned Insight Box */
    .insight-box {
        background-color: #f1f8ff;
        border-left: 5px solid #1976D2;
        padding: 1.25rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Info Box */
    .info-box {
        background-color: #f8f9fa;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
        }
        .subsection-header {
            font-size: 1.2rem;
        }
        .insight-box, .info-box {
            padding: 1rem;
        }
        .stDataFrame {
            font-size: 0.8rem;
        }
    }
    
    /* Custom Data Table Styling */
    .custom-dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    .custom-dataframe th {
        background-color: #1976D2;
        color: white;
        padding: 0.5rem;
        text-align: left;
    }
    .custom-dataframe td {
        padding: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .custom-dataframe tr:nth-child(even) {
        background-color: #f5f5f5;
    }
    
    /* Status Indicators */
    .status-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-negative {
        color: #F44336;
        font-weight: bold;
    }
    .status-neutral {
        color: #FF9800;
        font-weight: bold;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #1976D2;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f5f7f9;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-left: 16px;
        padding-right: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976D2 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Functions ---
def calculate_significance(conv_a, n_a, conv_b, n_b, test_type="one_tailed"):
    """
    Calculate statistical significance for A/B test
    
    Parameters:
    - conv_a: Conversions for control
    - n_a: Visitors for control
    - conv_b: Conversions for variant
    - n_b: Visitors for variant
    - test_type: "one_tailed" or "two_tailed"
    
    Returns:
    - confidence: Confidence level in percent
    - p_a: Conversion rate for control in percent
    - p_b: Conversion rate for variant in percent  
    - uplift: Percentage improvement
    - p_value: The p-value
    """
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    
    # Calculate p-value based on test type
    if test_type == "one_tailed":
        # One-tailed test (testing if variant is better)
        p_value = 1 - norm.cdf(z)
    else:
        # Two-tailed test (testing if there's any difference)
        p_value = 2 * (1 - norm.cdf(abs(z)))
        
    confidence = (1 - p_value) * 100
    uplift = ((p_b - p_a) / p_a) * 100
    return round(confidence, 2), round(p_a * 100, 2), round(p_b * 100, 2), round(uplift, 2), round(p_value, 4)

def calculate_required_conversions(n_a, conv_a, n_b, significance_level, test_type="one_tailed"):
    """
    Calculate the required conversions to reach significance
    
    Parameters:
    - n_a: Visitors for control
    - conv_a: Conversions for control
    - n_b: Visitors for variant
    - significance_level: The desired confidence level
    - test_type: "one_tailed" or "two_tailed"
    
    Returns:
    - Required number of conversions
    """
    z_scores = get_z_scores(test_type)
    
    if significance_level not in z_scores:
        return "Invalid significance level"
    
    z = z_scores[significance_level]
    p_a = conv_a / n_a  # Control conversion rate
    
    se = np.sqrt(p_a * (1 - p_a) * (2 / n_b))
    min_p_b = p_a + z * se
    required_conv_b = min_p_b * n_b
    
    return int(np.ceil(required_conv_b))

def get_z_scores(test_type="one_tailed"):
    """
    Get Z-scores based on test type
    """
    # Z-scores for one-tailed test
    one_tailed_z_scores = {
        80: 0.84,
        85: 1.04,
        90: 1.28,
        92: 1.41, 
        95: 1.645,
        97: 1.88,
        99: 2.33
    }
    
    # Z-scores for two-tailed test
    two_tailed_z_scores = {
        80: 1.28,
        85: 1.44,
        90: 1.645,
        92: 1.75,
        95: 1.96,
        97: 2.17,
        99: 2.58
    }
    
    return one_tailed_z_scores if test_type == "one_tailed" else two_tailed_z_scores

def calculate_test_duration(num_variants, baseline_cr, expected_uplift, daily_visitors, confidence_level, power_level, test_type="one_tailed"):
    """
    Calculate the required test duration in days
    
    Parameters:
    - num_variants: Number of variants including control
    - baseline_cr: Baseline conversion rate as a decimal
    - expected_uplift: Expected uplift as a decimal
    - daily_visitors: Average daily visitors
    - confidence_level: Confidence level (80, 85, 90, 92, 95, 97, 99)
    - power_level: Statistical power (80, 85, 90, 95)
    - test_type: "one_tailed" or "two_tailed"
    
    Returns:
    - Test duration in days
    """
    # Get z-score for alpha (confidence level)
    z_scores = get_z_scores(test_type)
    z_alpha = z_scores.get(confidence_level, 1.96)
    
    # Get z-score for beta (power)
    power_z_scores = {
        80: 0.84,
        85: 1.04,
        90: 1.28,
        95: 1.645
    }
    z_beta = power_z_scores.get(power_level, 0.84)
    
    # Calculate confidence constant
    confidence_constant = 2 * (z_alpha + z_beta)**2
    
    # Calculate test duration
    test_duration = (num_variants * confidence_constant * 
                    (np.sqrt(baseline_cr * (1 - baseline_cr)) / (baseline_cr * expected_uplift))**2) / daily_visitors
    
    return round(test_duration, 1)

def create_mde_chart(baseline_cr, daily_visitors, num_variants, confidence_level, power_level, test_type):
    """
    Create Minimum Detectable Effect (MDE) chart
    
    Parameters:
    - baseline_cr: Baseline conversion rate
    - daily_visitors: Daily visitors
    - num_variants: Number of variants including control
    - confidence_level: Confidence level
    - power_level: Statistical power
    - test_type: Test type (one-tailed or two-tailed)
    
    Returns:
    - Altair chart
    """
    # Define range of days and uplift values
    days_range = [7, 14, 21, 28, 42, 56, 84]
    uplift_values = []
    
    for days in days_range:
        # We need to solve for uplift given the test duration
        # Start with an initial guess
        uplift_guess = 0.05
        max_iterations = 20
        tolerance = 0.0001
        
        for _ in range(max_iterations):
            duration = calculate_test_duration(
                num_variants, 
                baseline_cr, 
                uplift_guess, 
                daily_visitors, 
                confidence_level, 
                power_level, 
                test_type
            )
            
            # If calculated duration is too high, increase uplift guess
            if duration > days:
                uplift_guess *= 1.1
            # If calculated duration is too low, decrease uplift guess
            elif duration < days:
                uplift_guess *= 0.9
            
            # Check if we've converged
            if abs(duration - days) < tolerance:
                break
        
        uplift_values.append(round(uplift_guess * 100, 1))
    
    # Create DataFrame for chart
    df = pd.DataFrame({
        'Days': days_range,
        'Minimum Detectable Effect (%)': uplift_values
    })
    
    # Create Altair chart
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Days:Q', title='Test Duration (Days)'),
        y=alt.Y('Minimum Detectable Effect (%):Q', title='Minimum Detectable Effect (%)'),
        tooltip=['Days', 'Minimum Detectable Effect (%)']
    ).properties(
        title=f'Minimum Detectable Effect Over Time',
        width=600,
        height=400
    ).configure_title(
        fontSize=16
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    return chart

# --- Main Layout ---
st.markdown('<p class="main-header">A/B Test Analyzer</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["🧪 Test Analyzer", "⏱️ Test Duration Calculator"])

# --- Test Analyzer Tab ---
with tab1:
    # --- Sidebar with improved styling ---
    with st.sidebar:
        st.markdown('<p class="subsection-header">Test Settings</p>', unsafe_allow_html=True)
        
        # Add test type selection
        test_type = st.radio(
            "Hypothesis Test Type",
            ["One-Tailed", "Two-Tailed"],
            index=0,  # Default to One-Tailed
            help="One-tailed tests if variant is better. Two-tailed tests if there's any difference."
        )
        test_type_value = "one_tailed" if test_type == "One-Tailed" else "two_tailed"
        
        num_variants = st.number_input("Number of Variants (including Control)", min_value=2, max_value=4, value=2, step=1)
        
        # Set significance threshold to fixed 95%
        significance_threshold = 95
        
        st.divider()
        with st.expander("About this tool", expanded=False):
            st.markdown("""
            This A/B test analyzer helps you determine if your test results are 
            statistically significant and provides insights on the number of 
            conversions needed to reach significance.
            
            - **Confidence Level**: How sure you can be that the results aren't due to chance
            - **Uplift**: The percentage improvement of the variant over control
            - **Required Conversions**: How many conversions needed to reach statistical significance
            """)
    
    st.markdown(f"""
    **Currently using: {test_type} Test** (95% significance threshold)

    **When to use each:**
    - Use **One-Tailed** when you're only interested in detecting improvements (common in marketing tests).
    - Use **Two-Tailed** when scientific validity is critical or when negative impacts must be detected.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Data Input Section ---
    st.markdown('<p class="section-header">Test Data Input</p>', unsafe_allow_html=True)

    data = []
    # Responsive columns design
    col_size = 12 // min(num_variants, 3)  # Dynamically adjust column widths
    cols = st.columns([col_size] * min(num_variants, 3))

    # Control data input
    with cols[0]:
        st.markdown('<p class="subsection-header">Control (A)</p>', unsafe_allow_html=True)
        control_visitors = st.number_input("Visitors", min_value=1, value=1000, key="visitors_0")
        control_conversions = st.number_input("Conversions", min_value=0, value=100, key="conversions_0")
        data.append((control_visitors, control_conversions))

    # Variant data inputs
    variant_idx = 1
    for i in range(1, num_variants):
        col_idx = i % len(cols)
        with cols[col_idx]:
            st.markdown(f'<p class="subsection-header">Variant {chr(65 + i)}</p>', unsafe_allow_html=True)
            visitors = st.number_input(f"Visitors", min_value=1, value=1000, key=f"visitors_{i}")
            conversions = st.number_input(f"Conversions", min_value=0, value=120, key=f"conversions_{i}")
            data.append((visitors, conversions))
            variant_idx += 1

    # Create a centered button with custom styling
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        analyze_button = st.button("Analyze Results", type="primary", use_container_width=True)

    # --- Results Section ---
    if analyze_button:
        st.markdown('<p class="section-header">Test Results</p>', unsafe_allow_html=True)
        
        control_visitors, control_conversions = data[0]
        control_rate = round((control_conversions / control_visitors) * 100, 2)
        
        # Prepare data for main results table
        rows = [
            {
                "Variant": "A (Control)",
                "Visitors": f"{control_visitors:,}",
                "Conversions": f"{control_conversions:,}",
                "Conversion Rate": f"{control_rate}%",
                "Uplift": "-",
                "Confidence": "-",
                "Significance": "-"
            }
        ]
        
        # Calculate results for each variant
        variant_results = []
        for i in range(1, num_variants):
            variant_visitors, variant_conversions = data[i]
            confidence, rate_a, rate_b, uplift, p_value = calculate_significance(
                control_conversions, control_visitors,
                variant_conversions, variant_visitors,
                test_type=test_type_value
            )
            
            is_significant = confidence >= significance_threshold
            significance_icon = "✅" if is_significant else "❌"
            significance_text = "Significant" if is_significant else "Not Significant"
            
            variant_results.append({
                "variant": chr(65 + i),
                "conversion_rate": rate_b,
                "uplift": uplift,
                "confidence": confidence,
                "is_significant": is_significant,
                "p_value": p_value
            })
            
            uplift_text = f"{uplift}% (+)" if uplift > 0 else f"{uplift}% (-)"
            
            rows.append({
                "Variant": f"{chr(65 + i)}",
                "Visitors": f"{variant_visitors:,}",
                "Conversions": f"{variant_conversions:,}",
                "Conversion Rate": f"{rate_b}%",
                "Uplift": uplift_text,
                "Confidence": f"{confidence}%",
                "Significance": f"{significance_icon} {significance_text}"
            })

        # Display main results table with improved styling
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            column_config={
                "Variant": st.column_config.TextColumn("Variant"),
                "Visitors": st.column_config.TextColumn("Visitors"),
                "Conversions": st.column_config.TextColumn("Conversions"),
                "Conversion Rate": st.column_config.TextColumn("Conv. Rate"),
                "Uplift": st.column_config.TextColumn("Uplift"),
                "Confidence": st.column_config.TextColumn("Confidence"),
                "Significance": st.column_config.TextColumn("Significance")
            },
            hide_index=True,
            use_container_width=True
        )
        
        
        # Insights Section
        st.markdown('<p class="section-header">Key Insights</p>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        
        if len(data) > 1:
            variant_b_conversions = data[1][1]
            variant_b_visitors = data[1][0]
            variant_b_rate = (variant_b_conversions / variant_b_visitors) * 100
            
            # Get the required conversions for the selected significance threshold
            required_threshold = calculate_required_conversions(
                control_visitors, 
                control_conversions, 
                variant_b_visitors,
                significance_threshold,
                test_type=test_type_value
            )
            
            more_needed_threshold = max(0, required_threshold - variant_b_conversions)
            
            if variant_results[0]["is_significant"]:
                significance_phrase = "better than" if test_type == "One-Tailed" else "different from"
                st.markdown(f"<p class='status-positive'>✅ <b>Variant B is statistically {significance_phrase} Control</b> with {variant_results[0]['confidence']}% confidence ({test_type} test).</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='status-positive'>📈 Conversion rate {'improved' if variant_results[0]['uplift'] > 0 else 'decreased'} by {abs(variant_results[0]['uplift'])}%.</p>", unsafe_allow_html=True)
            else:
                significance_phrase = "improvement over" if test_type == "One-Tailed" else "difference from"
                st.markdown(f"<p class='status-negative'>❌ <b>Not enough evidence</b> to declare statistical {significance_phrase} Control.</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='status-neutral'>⏳ Need {more_needed_threshold:,} more conversions to reach {significance_threshold}% confidence.</p>", unsafe_allow_html=True)
                
            # Show tips based on current results
            if variant_results[0]["uplift"] < 0:
                warning_text = "Variant B is performing worse than control." if test_type == "Two-Tailed" else "Variant B is performing worse than control, which contradicts the one-tailed test assumption."
                st.markdown(f"<p class='status-negative'>⚠️ <b>Warning:</b> {warning_text}</p>", unsafe_allow_html=True)
            elif variant_results[0]["confidence"] < 90:
                st.markdown("<p class='status-neutral'>💡 <b>Consider:</b> Continue the test to gather more data.</p>", unsafe_allow_html=True)
            
            # Add information about test type impact
            if test_type == "One-Tailed":
                two_tailed_conf, _, _, _, _ = calculate_significance(
                    control_conversions, control_visitors,
                    variant_b_conversions, variant_b_visitors,
                    test_type="two_tailed"
                )
                st.markdown(f"<p>🔄 With a <b>Two-Tailed</b> test, confidence would be {two_tailed_conf}%.</p>", unsafe_allow_html=True)
            else:
                one_tailed_conf, _, _, _, _ = calculate_significance(
                    control_conversions, control_visitors,
                    variant_b_conversions, variant_b_visitors,
                    test_type="one_tailed"
                )
                st.markdown(f"<p>🔄 With a <b>One-Tailed</b> test, confidence would be {one_tailed_conf}%.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p>Please add at least one variant besides control to see insights.</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Required Conversions Section
        st.markdown('<p class="section-header">Required Conversions Analysis</p>', unsafe_allow_html=True)
        
        if len(data) > 1:
            st.markdown('<p class="subsection-header">Conversions Required for Significance</p>', unsafe_allow_html=True)
            
            # Calculate required conversions for extended significance levels
            significance_levels = [80, 85, 90, 92, 95, 97, 99]
            required_conversions = []
            
            variant_visitors = data[1][0]  # Get visitors for variant B
            
            for level in significance_levels:
                required = calculate_required_conversions(
                    control_visitors, 
                    control_conversions, 
                    variant_visitors,
                    level,
                    test_type=test_type_value
                )
                current_conversions = data[1][1]  # Current conversions for variant B
                
                # Calculate how many more conversions are needed
                more_needed = max(0, required - current_conversions)
                
                required_conversions.append({
                    "Confidence Level": f"{level}%", 
                    "Required Conversions": f"{required:,}",
                    "Current Conversions": f"{current_conversions:,}",
                    "Additional Needed": f"{more_needed:,}",
                    "Required Conv. Rate": f"{round((required / variant_visitors) * 100, 2)}%"
                })

            required_df = pd.DataFrame(required_conversions)
            st.dataframe(required_df, hide_index=True, use_container_width=True)

# --- Test Duration Calculator Tab ---
with tab2:
    st.markdown('<p class="section-header">Test Duration Calculator</p>', unsafe_allow_html=True)
    st.markdown("""
    Calculate how long your A/B test needs to run to detect a statistically significant difference.
    This calculator uses the formula:
    
    **Test Duration (Days) = (Variants × ConfidenceConstant × (SQRT(CR×(1-CR))/(CR×Uplift))²) / DailyVisitors**
    
    Where **ConfidenceConstant = 2 × (Z_alpha + Z_beta)²**
    """)
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        test_type_duration = st.radio(
            "Test Type",
            ["One-Tailed", "Two-Tailed"],
            index=0,
            key="test_type_duration"
        )
        test_type_value_duration = "one_tailed" if test_type_duration == "One-Tailed" else "two_tailed"
        
        num_variants_duration = st.number_input(
            "Number of Variants (including Control)",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            key="num_variants_duration"
        )
        
        baseline_cr = st.number_input(
            "Baseline Conversion Rate (%)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            key="baseline_cr"
        ) / 100  # Convert to decimal
        
        expected_uplift = st.number_input(
            "Expected Uplift (%)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            key="expected_uplift"
        ) / 100  # Convert to decimal
    
    with col2:
        daily_visitors = st.number_input(
            "Daily Visitors (all variants combined)",
            min_value=1,
            max_value=1000000,
            value=500,
            step=10,
            key="daily_visitors"
        )
        
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[80, 85, 90, 92, 95, 97, 99],
            value=95,
            key="confidence_level"
        )
        
        power_level = st.select_slider(
            "Statistical Power",
            options=[80, 85, 90, 95],
            value=80,
            key="power_level"
        )
    
    # Calculate test duration
    test_duration = calculate_test_duration(
        num_variants_duration,
        baseline_cr,
        expected_uplift,
        daily_visitors,
        confidence_level,
        power_level,
        test_type_value_duration
    )
    
    # Display the result
    st.markdown('<p class="subsection-header">Test Duration Result</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h3 style="color: #1976D2; margin-top: 0;">Estimated Test Duration</h3>
            <p style="font-size: 2rem; font-weight: 700; margin: 0;">{test_duration} days</p>
            <p style="font-size: 1rem; color: #666; margin-top: 0.5rem;">
                {test_duration // 7} weeks, {test_duration % 7} days
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3 style="color: #4CAF50; margin-top: 0;">Test Parameters</h3>
            <ul style="list-style-type: none; padding-left: 0; margin-top: 0.5rem;">
                <li>• Number of variants: {num_variants_duration}</li>
                <li>• Baseline conversion rate: {baseline_cr_percent}%</li>
                <li>• Expected uplift: {expected_uplift_percent}%</li>
                <li>• Daily visitors: {daily_visitors:,}</li>
                <li>• Confidence level: {confidence_level}%</li>
                <li>• Statistical power: {power_level}%</li>
                <li>• Test type: {test_type_duration}</li>
            </ul>
        </div>
        """.format(
            num_variants_duration=num_variants_duration,
            baseline_cr_percent=round(baseline_cr * 100, 2),
            expected_uplift_percent=round(expected_uplift * 100, 2),
            daily_visitors=daily_visitors,
            confidence_level=confidence_level,
            power_level=power_level,
            test_type_duration=test_type_duration
        ), unsafe_allow_html=True)
    
    # Minimum Detectable Effect Chart
    st.markdown('<p class="subsection-header">Minimum Detectable Effect (MDE) Chart</p>', unsafe_allow_html=True)
    st.markdown("""
    This chart shows the minimum uplift percentage that can be detected with statistical significance based on test duration.
    """)
    
    mde_chart = create_mde_chart(
        baseline_cr, 
        daily_visitors, 
        num_variants_duration, 
        confidence_level, 
        power_level, 
        test_type_value_duration
    )
    st.altair_chart(mde_chart, use_container_width=True)
    
    with st.expander("Understanding Minimum Detectable Effect", expanded=False):
        st.markdown("""
        ### What is Minimum Detectable Effect (MDE)?
        
        The Minimum Detectable Effect (MDE) is the smallest improvement in your conversion rate that your test can reliably detect with your chosen confidence level and statistical power.

        ### How to use this chart:
        
        - **Reading the chart:** For any point on the line, the y-axis value shows the minimum uplift percentage you can detect if you run your test for the number of days shown on the x-axis.
        - **Planning your test:** If you expect a 5% uplift, find the 5% on the y-axis and see how many days you'll need to run the test.
        - **Being realistic:** If the MDE for reasonable test durations (e.g., 2-4 weeks) is much higher than the expected uplift from your change, you may need to:
          1. Increase your sample size (more traffic)
          2. Target a higher-volume segment
          3. Aim for bigger improvements in your design
        
        ### Example:
        If the chart shows a 15% MDE at 14 days, it means you need at least a 15% improvement in conversion rate to detect it with statistical significance in a 14-day test.
        """)

# Add a section for helpful information and FAQ
with tab2:
    st.markdown('<p class="section-header">Tips & Best Practices</p>', unsafe_allow_html=True)
    
    with st.expander("How can I reduce the required test duration?", expanded=False):
        st.markdown("""
        ### Strategies to reduce test duration:

        1. **Increase daily traffic:** More visitors = faster results
        2. **Focus on higher-converting segments:** Higher baseline conversion rates require smaller sample sizes
        3. **Test bigger changes:** Larger expected uplifts can be detected faster
        4. **Reduce the number of variants:** More variants require more time
        5. **Accept a lower confidence level:** 90% instead of 95% requires less time (but increases risk)
        6. **Lower your statistical power:** 80% instead of 90% reduces time (but increases chance of missing real effects)
        7. **Use one-tailed tests:** If you're only interested in improvements, not detecting negative impacts
        """)
    
    with st.expander("How to interpret A/B test results?", expanded=False):
        st.markdown("""
        ### Keys to proper interpretation:

        1. **Don't end tests early:** Stopping when you see "significance" before your planned duration leads to false positives
        2. **Consider practical significance:** A statistically significant 0.1% uplift may not be worth implementing
        3. **Look for segment differences:** The variant might perform differently across user segments
        4. **Consider long-term effects:** Some changes have different short vs. long-term impacts
        5. **Run follow-up tests:** Validate important findings with additional tests
        6. **Look beyond just conversions:** Consider engagement, revenue per user, user satisfaction
        7. **Context matters:** Business cycles, seasonality, and external events can impact results
        """)
        
    with st.expander("Statistical terminology explained", expanded=False):
        st.markdown("""
        ### Key Statistical Terms:

        - **Statistical Significance:** The probability that the observed difference between variants is not due to random chance
        - **Confidence Level:** How certain you are that the observed results are real (e.g., 95% confidence)
        - **Statistical Power:** The probability of detecting a true effect when it exists (e.g., 80% power)
        - **Type I Error:** False positive - concluding there's an effect when there isn't one
        - **Type II Error:** False negative - failing to detect an effect that actually exists
        - **P-value:** The probability of obtaining results at least as extreme as those observed, assuming the null hypothesis is true
        - **Z-score:** The number of standard deviations a data point is from the mean
        """)

# Add JavaScript code to detect screen size and set a session state variable
# This helps with responsive design
st.markdown("""
<script>
    // Set screen width in session state for responsive design
    const screenWidth = window.innerWidth;
    const key = 'screen_width';
    const value = screenWidth;
    
    // Using setTimeout to ensure this runs after Streamlit is fully loaded
    setTimeout(() => {
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            key: key,
            value: value,
        }, "*");
    }, 100);
    
    // Listen for window resize events
    window.addEventListener('resize', function() {
        const newWidth = window.innerWidth;
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            key: key,
            value: newWidth,
        }, "*");
    });
</script>
""", unsafe_allow_html=True)
