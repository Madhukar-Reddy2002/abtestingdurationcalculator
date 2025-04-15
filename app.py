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
        font-weight: 800;
        color: #1565C0;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        background: linear-gradient(90deg, #1565C0, #64B5F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
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
        .insight-box {
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

# --- Sidebar with improved styling ---
with st.sidebar:
    st.markdown('<p class="subsection-header">Test Settings</p>', unsafe_allow_html=True)
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
    
    # Visualization with improved styling
    if len(data) > 1:
        st.markdown('<p class="section-header">Visualization</p>', unsafe_allow_html=True)
        
        # Conversion Rate Chart
        chart_data = pd.DataFrame({
            'Variant': [row["Variant"] for row in rows],
            'Conversion Rate': [float(row["Conversion Rate"].replace("%", "")) for row in rows]
        })
        
        # Enhanced styling for the chart
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
            title={
                'text': 'Conversion Rate by Variant',
                'fontSize': 20,
                'fontWeight': 'bold',
                'color': '#1976D2'
            },
            width='container',
            height=300
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            titleFontWeight='bold'
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        st.altair_chart(conversion_chart, use_container_width=True)
    
    # Required Conversions Section with improved responsiveness
    st.markdown('<p class="section-header">Required Conversions Analysis</p>', unsafe_allow_html=True)
    
    # Use a dynamic column layout that works better on mobile
    if st.session_state.get('screen_width', 1000) < 768:
        # Stack columns on mobile
        cols = st.columns([1])
        insights_col = cols[0]
        required_col = cols[0]
    else:
        # Side by side on desktop
        required_col, insights_col = st.columns([3, 2])
    
    with required_col:
        if len(data) > 1:
            st.markdown('<p class="subsection-header">Conversions Required for Significance</p>', unsafe_allow_html=True)
            
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
            st.dataframe(required_df, hide_index=True, use_container_width=True)
    
    with insights_col:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown('<p class="subsection-header">Key Insights</p>', unsafe_allow_html=True)
        
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
                st.markdown(f"<p class='status-positive'>✅ <b>Variant B shows statistically significant improvement</b> with {variant_results[0]['confidence']}% confidence.</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='status-positive'>📈 Conversion rate improved by {variant_results[0]['uplift']}%.</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='status-negative'>❌ <b>Not enough evidence</b> to declare Variant B better.</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='status-neutral'>⏳ Need {more_needed_95:,} more conversions to reach 95% confidence.</p>", unsafe_allow_html=True)
                
            # Show tips based on current results
            if variant_results[0]["uplift"] < 0:
                st.markdown("<p class='status-negative'>⚠️ <b>Warning:</b> Variant B is performing worse than control.</p>", unsafe_allow_html=True)
            elif variant_results[0]["confidence"] < 90:
                st.markdown("<p class='status-neutral'>💡 <b>Consider:</b> Continue the test to gather more data.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p>Please add at least one variant besides control to see insights.</p>", unsafe_allow_html=True)
        
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
