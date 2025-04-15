import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Set page config
st.set_page_config(page_title="A/B Test Analyzer", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1565C0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1976D2;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f1f8ff;
        border-left: 4px solid #1976D2;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Functions ---
def calculate_significance(conv_a, n_a, conv_b, n_b):
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test
    confidence = (1 - p_value) * 100
    uplift = ((p_b - p_a) / p_a) * 100
    return round(confidence, 2), round(p_a * 100, 2), round(p_b * 100, 2), round(uplift, 2), round(p_value, 4)

def calculate_required_conversions(n_a, conv_a, n_b, significance_level):
    z_scores = {
        85: 1.44,
        90: 1.645,
        92: 1.75,
        95: 1.96
    }
    
    if significance_level not in z_scores:
        return "Invalid significance level"
    
    z = z_scores[significance_level]
    p_a = conv_a / n_a  # Control conversion rate
    
    se = np.sqrt(p_a * (1 - p_a) * (2 / n_b))
    min_p_b = p_a + z * se
    required_conv_b = min_p_b * n_b
    
    return int(np.ceil(required_conv_b))

# --- Main Layout ---
st.markdown('<p class="main-header">A/B Test Analyzer</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("Test Settings")
    num_variants = st.number_input("Number of Variants (including Control)", min_value=2, max_value=4, value=2, step=1)
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

# --- Data Input Section ---
st.markdown('<p class="section-header">Test Data Input</p>', unsafe_allow_html=True)

data = []
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Control (A)")
    control_visitors = st.number_input("Visitors", min_value=1, value=1000, key="visitors_0")
    control_conversions = st.number_input("Conversions", min_value=0, value=100, key="conversions_0")
    data.append((control_visitors, control_conversions))

with col2:
    for i in range(1, num_variants):
        st.subheader(f"Variant {chr(65 + i)}")
        visitors = st.number_input(f"Visitors", min_value=1, value=1000, key=f"visitors_{i}")
        conversions = st.number_input(f"Conversions", min_value=0, value=120, key=f"conversions_{i}")
        data.append((visitors, conversions))

analyze_button = st.button("Analyze Results", type="primary")

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
            variant_conversions, variant_visitors
        )
        
        is_significant = confidence >= 95
        significance_icon = "✅" if is_significant else "❌"
        significance_text = "Significant" if is_significant else "Not Significant"
        
        variant_results.append({
            "variant": chr(65 + i),
            "conversion_rate": rate_b,
            "uplift": uplift,
            "confidence": confidence,
            "is_significant": is_significant
        })
        
        rows.append({
            "Variant": f"{chr(65 + i)}",
            "Visitors": f"{variant_visitors:,}",
            "Conversions": f"{variant_conversions:,}",
            "Conversion Rate": f"{rate_b}%",
            "Uplift": f"{uplift}%" + (" (+" if uplift > 0 else " ("),
            "Confidence": f"{confidence}%",
            "Significance": f"{significance_icon} {significance_text}"
        })

    # Display main results table
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
    )
    
    # Visualization
    if len(data) > 1:
        st.markdown('<p class="section-header">Visualization</p>', unsafe_allow_html=True)
        
        # Conversion Rate Chart
        chart_data = pd.DataFrame({
            'Variant': [row["Variant"] for row in rows],
            'Conversion Rate': [float(row["Conversion Rate"].replace("%", "")) for row in rows]
        })
        
        conversion_chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Variant', sort=None, title='Variant'),
            y=alt.Y('Conversion Rate', title='Conversion Rate (%)'),
            color=alt.condition(
                alt.datum.Variant == 'A (Control)',
                alt.value('#1976D2'),  # Control color
                alt.value('#4CAF50')   # Variant color
            ),
            tooltip=['Variant', 'Conversion Rate']
        ).properties(
            title='Conversion Rate by Variant',
            width=500,
            height=300
        )
        
        st.altair_chart(conversion_chart, use_container_width=True)
    
    # Required Conversions Section
    st.markdown('<p class="section-header">Required Conversions Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if len(data) > 1:
            # Calculate required conversions for each significance level
            significance_levels = [85, 90, 92, 95]
            required_conversions = []
            
            variant_visitors = data[1][0]  # Get visitors for variant B
            
            for level in significance_levels:
                required = calculate_required_conversions(
                    control_visitors, 
                    control_conversions, 
                    variant_visitors,
                    level
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
            st.dataframe(required_df, hide_index=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### Key Insights")
        
        if len(data) > 1:
            variant_b_conversions = data[1][1]
            variant_b_visitors = data[1][0]
            variant_b_rate = (variant_b_conversions / variant_b_visitors) * 100
            
            # Get the 95% confidence required conversions
            required_95 = calculate_required_conversions(
                control_visitors, 
                control_conversions, 
                variant_b_visitors,
                95
            )
            
            more_needed_95 = max(0, required_95 - variant_b_conversions)
            
            if variant_results[0]["is_significant"]:
                st.markdown(f"✅ **Variant B shows statistically significant improvement** with {variant_results[0]['confidence']}% confidence.")
                st.markdown(f"📈 Conversion rate improved by {variant_results[0]['uplift']}%.")
            else:
                st.markdown(f"❌ **Not enough evidence** to declare Variant B better.")
                st.markdown(f"⏳ Need {more_needed_95:,} more conversions to reach 95% confidence.")
                
            # Show tips based on current results
            if variant_results[0]["uplift"] < 0:
                st.markdown("⚠️ **Warning:** Variant B is performing worse than control.")
            elif variant_results[0]["confidence"] < 90:
                st.markdown("💡 **Consider:** Continue the test to gather more data.")
        else:
            st.markdown("Please add at least one variant besides control to see insights.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Information Sections (All Collapsed by Default)
    with st.expander("How Statistical Significance Works", expanded=False):
        st.markdown("""
        Statistical significance in A/B testing determines whether the observed differences between variants are real or just due to random chance.

        ### Key Concepts:

        - **Conversion Rate:** The percentage of visitors who complete the desired action
        - **Statistical Confidence:** The probability that the observed difference is not due to chance
        - **P-value:** The probability of seeing the observed results (or more extreme) if there was no real difference
        - **Significance Threshold:** Typically set at 95% confidence (p-value < 0.05)

        ### The Calculation:

        We use a Z-test for proportions:
        1. Calculate the conversion rates for control and variant
        2. Determine the pooled standard error
        3. Calculate the Z-score based on the difference in rates
        4. Convert to a confidence level or p-value
        """)

    with st.expander("How Required Conversions Are Calculated", expanded=False):
        st.markdown("""
        The "Required Conversions" analysis shows how many conversions your variant needs to reach statistical significance.

        ### Assumptions:
        
        - Control conversion rate remains constant
        - Variant visitor count remains constant
        - We're detecting a positive improvement

        ### The Process:
        
        1. We use the Z-score for the desired confidence level
        2. Calculate the standard error based on the control conversion rate
        3. Determine the minimum conversion rate needed to achieve significance
        4. Convert this rate to the number of conversions needed
        """)

    with st.expander("Best Practices for A/B Testing", expanded=False):
        st.markdown("""
        ### Tips for Effective A/B Testing:

        - **Plan your sample size** before starting the test
        - **Run tests for complete business cycles** (minimum one week, preferably 2+ weeks)
        - **Wait for 95% confidence** before declaring a winner
        - **Check for external factors** that might influence results
        - **Test one change at a time** for clear cause-effect relationships
        - **Consider practical significance** alongside statistical significance
        - **Segment your results** to discover if the change affects some user groups differently
        """)
