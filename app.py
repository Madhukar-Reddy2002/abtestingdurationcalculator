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
num_variants = st.sidebar.number_input("Number of Variants (including Control)", min_value=2, max_value=4, value=2, step=1)

st.title("📊 A/B Test Statistical Significance Calculator")
st.markdown("""Enter the number of visitors and conversions for each variant below. 
We'll calculate conversion rates, uplift, and statistical significance.
""")

# --- Data Input ---
data = []

for i in range(num_variants):
    st.subheader(f"Variant {'A (Control)' if i == 0 else chr(65 + i)}")
    col1, col2 = st.columns(2)
    with col1:
        visitors = st.number_input(f"Enter visitors for Variant {'A' if i == 0 else chr(65 + i)}", min_value=1, key=f"visitors_{i}")
    with col2:
        conversions = st.number_input(f"Enter conversions for Variant {'A' if i == 0 else chr(65 + i)}", min_value=0, key=f"conversions_{i}")
    data.append((visitors, conversions))

# --- Results ---
if st.button("Calculate"):
    st.header("📈 Results")
    control_visitors, control_conversions = data[0]
    control_rate = round((control_conversions / control_visitors) * 100, 2)

    results = []

    for i in range(1, num_variants):
        variant_visitors, variant_conversions = data[i]
        confidence, rate_a, rate_b, uplift = calculate_significance(
            control_conversions, control_visitors,
            variant_conversions, variant_visitors
        )
        results.append({
            "Variant": f"{chr(65 + i)}",
            "Control Visitors": control_visitors,
            "Control Conversions": control_conversions,
            "Control Rate (%)": rate_a,
            "Variant Visitors": variant_visitors,
            "Variant Conversions": variant_conversions,
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
        "Conversion Rate (%)": [control_rate] + [r["Variant Rate (%)"] for r in results]
    })

    fig = px.bar(chart_data, x="Variant", y="Conversion Rate (%)", color="Variant", text="Conversion Rate (%)")
    st.plotly_chart(fig)

    # Explanation of formula
    with st.expander("📘 How We Calculate Significance (Click to Learn)"):
        st.markdown("""
        We use a **Z-test for proportions**, a classic statistics method to compare whether your A/B test results are meaningfully different or just due to randomness.

        #### 🧮 The Math Made Simple:

        1. **Average rate (pooled)**:
           $$ p_{pool} = \frac{conversions_A + conversions_B}{visitors_A + visitors_B} $$

        2. **Standard Error (SE)**: like a "noise level" between the two groups
           $$ SE = \sqrt{p_{pool}(1 - p_{pool})\left(\frac{1}{visitors_A} + \frac{1}{visitors_B}\right)} $$

        3. **Z-score** = how many SEs away the difference is
           $$ Z = \frac{rate_B - rate_A}{SE} $$

        4. **Confidence %** is derived from the Z-score.

        A result is **statistically significant** when confidence ≥ 95%. That means there's less than 5% chance it's due to luck!
        """)

    with st.expander("💡 A/B Testing Tips"):
        st.markdown("""
        - ✅ Aim for **95% confidence** or more before acting on results.
        - 📈 Bigger samples → more reliable results.
        - 🕐 Don’t end tests early — wait for data!
        - ⚖️ A small improvement with high confidence is more valuable than a huge one with low certainty.
        """)
