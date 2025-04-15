import streamlit as st
import pandas as pd
import math
from scipy import stats

st.set_page_config(page_title="A/B Test Significance Calculator", layout="centered")

st.title("\U0001F4CA A/B Test Significance Calculator")
st.write("This calculator uses industry-standard methods to test the statistical significance of A/B test results.")

st.sidebar.header("Input Parameters")

control_visitors = st.sidebar.number_input("Control Group Visitors", min_value=1, value=1000)
control_conversions = st.sidebar.number_input("Control Group Conversions", min_value=0, value=100)
variant_visitors = st.sidebar.number_input("Variant Group Visitors", min_value=1, value=1000)
variant_conversions = st.sidebar.number_input("Variant Group Conversions", min_value=0, value=130)

test_type = st.sidebar.radio("Test Type", ["Two-tailed", "One-tailed"])
alpha = st.sidebar.slider("Significance Level (\u03b1)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

# Function using pooled standard error (aligned with Neil Patel calculator)
def calculate_ab_significance(c_success, c_total, v_success, v_total, alpha=0.05, one_tailed=False):
    p1 = c_success / c_total
    p2 = v_success / v_total
    p_pool = (c_success + v_success) / (c_total + v_total)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / c_total + 1 / v_total))
    z = (p2 - p1) / se if se > 0 else 0

    if one_tailed:
        p_value = 1 - stats.norm.cdf(z) if z > 0 else 1.0
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    ci_range = stats.norm.ppf(1 - alpha / 2) * se
    ci_lower = (p2 - p1) - ci_range
    ci_upper = (p2 - p1) + ci_range
    significance = p_value < alpha

    return {
        "control_rate": p1 * 100,
        "variant_rate": p2 * 100,
        "uplift": (p2 - p1) * 100,
        "z_score": z,
        "p_value": p_value,
        "significant": significance,
        "ci_lower": ci_lower * 100,
        "ci_upper": ci_upper * 100
    }

if st.button("Calculate Significance"):
    result = calculate_ab_significance(
        control_conversions,
        control_visitors,
        variant_conversions,
        variant_visitors,
        alpha=alpha,
        one_tailed=(test_type == "One-tailed")
    )

    st.subheader("Results")

    col1, col2 = st.columns(2)
    col1.metric("Control Conversion Rate", f"{result['control_rate']:.2f}%")
    col2.metric("Variant Conversion Rate", f"{result['variant_rate']:.2f}%")

    col1.metric("Uplift", f"{result['uplift']:.2f}%")
    col2.metric("P-Value", f"{result['p_value']:.4f}")

    ci_text = f"{result['ci_lower']:.2f}% to {result['ci_upper']:.2f}%"
    st.write(f"**95% Confidence Interval:** {ci_text}")

    if result['significant']:
        st.success("Your result is statistically significant. The observed uplift is unlikely due to chance.")
    else:
        st.error("Your result is not statistically significant. You may need more data or there may be no real effect.")
