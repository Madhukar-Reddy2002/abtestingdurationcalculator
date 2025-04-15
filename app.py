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
st.markdown("""
Enter the number of visitors and conversions for each variant below. 
We'll calculate conversion rates, uplift, and statistical significance.
""")

# --- Data Input ---
data = []

for i in range(num_variants):
    st.subheader(f"Variant {'A (Control)' if i == 0 else chr(65 + i)}")
    col1, col2 = st.columns(2)
    with col1:
        visitors = st.number_input(f"Visitors for Variant {'A' if i == 0 else chr(65 + i)}", min_value=1, key=f"visitors_{i}")
    with col2:
        conversions = st.number_input(f"Conversions for Variant {'A' if i == 0 else chr(65 + i)}", min_value=0, key=f"conversions_{i}")
    data.append((visitors, conversions))

# --- Results ---
if st.button("Calculate"):
    st.header("📈 Results")
    control_visitors, control_conversions = data[0]
    control_rate = round((control_conversions / control_visitors) * 100, 2)

    rows = [
        {
            "Variant": "A (Control)",
            "Visitors": control_visitors,
            "Conversions": control_conversions,
            "Conversion Rate (%)": control_rate,
            "Uplift (%)": "-",
            "Confidence (%)": "-",
            "Statistically Significant": "-"
        }
    ]

    for i in range(1, num_variants):
        variant_visitors, variant_conversions = data[i]
        confidence, rate_a, rate_b, uplift = calculate_significance(
            control_conversions, control_visitors,
            variant_conversions, variant_visitors
        )
        rows.append({
            "Variant": f"{chr(65 + i)}",
            "Visitors": variant_visitors,
            "Conversions": variant_conversions,
            "Conversion Rate (%)": rate_b,
            "Uplift (%)": uplift,
            "Confidence (%)": confidence,
            "Statistically Significant": "✅ Yes" if confidence >= 95 else "❌ No"
        })

    df = pd.DataFrame(rows)
    st.dataframe(df)

    # Bar chart of conversion rates
    chart_data = pd.DataFrame({
        "Variant": [row["Variant"] for row in rows],
        "Conversion Rate (%)": [row["Conversion Rate (%)"] for row in rows]
    })

    fig = px.bar(chart_data, x="Variant", y="Conversion Rate (%)", color="Variant", text="Conversion Rate (%)")
    st.plotly_chart(fig)

    # Explanation of formula (shown by default)
    with st.expander("📘 How Do We Calculate Significance? (Open to Learn More)", expanded=True):
        st.markdown("""
        We use a **Z-test for proportions** – a statistical method that tells us whether the difference between two conversion rates is **real or just due to random chance**.

        ### 🔍 The Logic Behind It:

        - Imagine you're flipping a coin. If you got 60 heads out of 100, is that lucky or a sign it's not fair?
        - Same here! We compare the control and variant rates to see if the difference is big **enough** to not be just chance.

        ### 📐 The Formula (Simplified):

        1. **Pooled Rate (combined average):**
           $$ p_{pool} = \frac{conversions_A + conversions_B}{visitors_A + visitors_B} $$

        2. **Standard Error (SE):** like the expected "wiggle room"
           $$ SE = \sqrt{p_{pool}(1 - p_{pool})\left(\frac{1}{visitors_A} + \frac{1}{visitors_B}\right)} $$

        3. **Z-score:**
           $$ Z = \frac{rate_B - rate_A}{SE} $$

        4. **Confidence (%):**
           - Based on how extreme the Z-score is.
           - **Higher confidence = more likely the uplift is real!**

        If confidence ≥ 95%, you can be pretty sure your result is significant! ✅
        """)

    with st.expander("💡 Extra Tips for Better A/B Testing"):
        st.markdown("""
        - ✅ **Wait for enough data** before making decisions.
        - 🎯 Try reaching **at least 95% confidence**.
        - 🧠 Don’t focus only on uplift — check sample sizes and error margins too!
        - 📊 Visuals help, but always trust the numbers!
        """)
