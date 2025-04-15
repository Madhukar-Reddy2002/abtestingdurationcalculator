import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.express as px

# --- Function to calculate significance ---
def calculate_significance(conv_a, n_a, conv_b, n_b):
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_value = 1 - norm.cdf(abs(z))
    confidence = (1 - p_value) * 100
    uplift = ((p_b - p_a) / p_a) * 100
    return round(confidence, 2), round(p_a * 100, 2), round(p_b * 100, 2), round(uplift, 2)

# --- Sidebar ---
st.sidebar.title("A/B Test Settings")
num_variants = st.sidebar.slider("Number of Variants (including Control)", 2, 4, 2)
show_formula = st.sidebar.checkbox("Show Statistical Explanation")
show_tips = st.sidebar.checkbox("Show Tips")

st.title("📊 A/B Test Statistical Significance Calculator")
st.markdown("""Enter the number of visitors and conversions for each variant below. 
We'll calculate conversion rates, uplift, and statistical significance.
""")

# --- Data Input ---
data = []

for i in range(num_variants):
    st.subheader(f"Variant {'A (Control)' if i == 0 else chr(66 + i - 1)}")
    col1, col2 = st.columns(2)
    with col1:
        visitors = st.number_input(f"Visitors - Variant {i+1}", min_value=1, key=f"visitors_{i}")
    with col2:
        conversions = st.number_input(f"Conversions - Variant {i+1}", min_value=0, key=f"conversions_{i}")
    data.append((visitors, conversions))

# --- Results ---
if st.button("Calculate"):
    st.header("📈 Results")
    control_visitors, control_conversions = data[0]
    results = []

    for i in range(1, num_variants):
        variant_visitors, variant_conversions = data[i]
        confidence, rate_a, rate_b, uplift = calculate_significance(
            control_conversions, control_visitors,
            variant_conversions, variant_visitors
        )
        results.append({
            "Variant": f"{'A (Control)' if i == 0 else chr(66 + i - 1)}",
            "Control Rate (%)": rate_a,
            "Variant Rate (%)": rate_b,
            "Uplift (%)": uplift,
            "Confidence (%)": confidence,
            "Statistically Significant": "✅ Yes" if confidence >= 95 else "❌ No"
        })

    df = pd.DataFrame(results)
    st.dataframe(df)

    # Bar chart of conversion rates
    chart_data = pd.DataFrame({
        "Variant": ["A (Control)"] + [r["Variant"] for r in results],
        "Conversion Rate (%)": [df["Control Rate (%)"].iloc[0]] + [r["Variant Rate (%)"] for r in results]
    })

    fig = px.bar(chart_data, x="Variant", y="Conversion Rate (%)", color="Variant", text="Conversion Rate (%)")
    st.plotly_chart(fig)

    # Explanation of formula
    if show_formula:
        with st.expander("🧠 How We Calculate Significance"):
            st.markdown("""
            **We use a Z-test for proportions.** Here's the breakdown:

            1. Calculate the pooled conversion rate:
            $$ p_{pool} = \frac{conversions_A + conversions_B}{visitors_A + visitors_B} $$

            2. Standard error:
            $$ SE = \sqrt{p_{pool}(1 - p_{pool})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)} $$

            3. Z-score:
            $$ Z = \frac{p_B - p_A}{SE} $$

            4. Confidence:
            $$ Confidence = (1 - p\_value) * 100 $$
            
            This tells us how likely it is that the difference in conversion rates is real.
            """)

    if show_tips:
        with st.expander("💡 Tips for Better A/B Testing"):
            st.markdown("""
            - Aim for **95%+ confidence** to make reliable decisions.
            - **Run longer tests** if your sample size is small.
            - Track **uplift (%)** to measure improvement over the control.
            - Consider external factors like time of day, user behavior changes, etc.
            """)
