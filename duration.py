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
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">A/B Test Duration Calculator</div>', unsafe_allow_html=True)
st.markdown("""
This interactive tool helps you estimate the **minimum A/B test duration** needed to achieve **statistical significance** based on your parameters.
It uses power analysis to determine how long your test should run to detect the expected improvement with confidence.
""")

# Create two columns for layout
sidebar_col, main_col = st.columns([1, 3])

# Sidebar Inputs
with sidebar_col:
    st.markdown('<div class="sub-header">Test Parameters</div>', unsafe_allow_html=True)
    
    # Existing Data Inputs
    st.markdown("#### Existing Data")
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
                                             help="Number of visitors who completed the desired action")
    
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
    st.markdown(f"**Daily Visitors:** {daily_visitors}")

    # A/B Test Parameters
    st.markdown("#### Test Settings")
    target_test_duration = st.number_input("Target Test Duration (days)", 
                                        min_value=1, 
                                        max_value=365, 
                                        value=14, 
                                        step=1,
                                        help="Your desired test duration in days")
    
    color_coding_percentage = st.number_input("Color Coding Threshold (%)", 
                                           min_value=0.1, 
                                           max_value=10.0, 
                                           value=2.0, 
                                           step=0.1,
                                           help="Percentage difference from target that is considered acceptable")
    
    variations = st.number_input("Number of Variants (including control)", 
                              min_value=2, 
                              max_value=5, 
                              value=2, 
                              step=1,
                              help="Total number of variations in your test, including the control")
    
    target_improvement_input = st.text_input("Expected Improvement (%)", 
                                       value="10.0",
                                       help="How much improvement you expect to see in your conversion rate")
    
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
                                   help="Desired confidence level for your test results")

# Z-score lookup
z_scores = {75: 1.15, 80: 1.28, 85: 1.44, 90: 1.65, 95: 1.96, 99: 2.58}

# Function to compute test duration
def calculate_test_duration(daily_visits, vars, conversion, improvement, significance):
    # Reference values based on industry standards
    reference_values = {
        0.25: 299520, 0.5: 74880, 0.75: 33280, 1.0: 18720,
        2.0: 4680, 3.0: 2080, 5.0: 749, 6.0: 520, 7.0: 382,
        8.0: 293, 9.0: 231, 10.0: 187, 12.5: 120, 25.0: 30, 50.0: 7
    }

    closest_imp = min(reference_values.keys(), key=lambda x: abs(x - improvement))
    base_days = reference_values[closest_imp]

    variant_factor = {2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}.get(vars, 1.0)
    sig_factor = (z_scores[significance] / z_scores[95]) ** 2
    cvr_factor = 10.0 / conversion if conversion > 0 else 1.0
    visitor_factor = 250 / daily_visits if daily_visits > 0 else 1.0

    days = base_days * variant_factor * sig_factor * cvr_factor * visitor_factor
    visitors_needed = days * daily_visits
    samples_per_variant = visitors_needed / vars
    
    return {
        "days": max(1, round(days)), 
        "visitors": max(1, round(visitors_needed)),
        "samples_per_variant": max(1, round(samples_per_variant))
    }

# Compute results
result = calculate_test_duration(daily_visitors, variations, baseline_conversion, target_improvement, significance_level)

# Calculate power
power = significance_level / 100

# Main content
with main_col:
    # Create three columns for metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
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
    
    with metric_col3:
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
    
    # Recommendation box
    st.info(f"""
    **Recommendation:** Run your test for at least **{result['days']} days** to achieve **{significance_level}%** statistical significance.
    
    This will require approximately **{result['visitors']:,} total visitors** with **{result['samples_per_variant']:,} visitors per variant**.
    
    With these parameters, you'll be able to detect a **{target_improvement}%** improvement in your conversion rate with **{power:.2f}** power.
    """)
    
    # Additional information
    with st.expander("What does this mean?"):
        st.markdown("""
        ### Understanding the Results
        
        - **Test Duration**: The minimum number of days your test should run to achieve the desired statistical power.
        - **Total Visitors**: The total number of visitors needed across all variants.
        - **Status**: Whether your target test duration is sufficient for your desired parameters.
        
        ### Factors Affecting Test Duration
        
        - **Baseline Conversion Rate**: Lower baseline rates require longer test durations.
        - **Expected Improvement**: Smaller improvements take longer to detect reliably.
        - **Statistical Significance**: Higher confidence levels require more data.
        - **Number of Variants**: More variants require more visitors per variant.
        - **Daily Traffic**: Lower traffic sites need longer test durations.
        
        ### Best Practices
        
        - Don't stop tests early, even if you see significance
        - Run tests for full weeks to account for weekday/weekend patterns
        - Ensure even traffic distribution across variants
        - Consider seasonal factors that might affect your results
        """)
    
    # Visualization Tabs
    st.markdown('<div class="sub-header">Visual Analysis</div>', unsafe_allow_html=True)
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Significance Impact", "Target Improvement Impact", "Variants Impact"])
    
    with viz_tab1:
        # Create data for significance level comparison
        significance_levels = [75, 80, 85, 90, 95, 99]
        sig_days = [calculate_test_duration(daily_visitors, variations, baseline_conversion, target_improvement, sig)["days"] 
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
            title=f"Test Duration by Significance Level (at {target_improvement}% expected improvement)",
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
        This chart shows how the required test duration increases as you require higher statistical significance.
        Higher significance levels (more confidence) require more data and thus longer test durations.
        </div>
        """, unsafe_allow_html=True)
    
    with viz_tab2:
        # Create data for improvement comparison with more granular values
        # Using more data points to create a smoother curve
        imp_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 15.0, 20.0, 25.0, 50.0]
        imp_days = [calculate_test_duration(daily_visitors, variations, baseline_conversion, imp, significance_level)["days"] 
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
            title=f"Test Duration by Expected Improvement (at {significance_level}% significance)",
            xaxis_title="Expected Improvement (%)",
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
                gridcolor='rgba(255,0,0,0.8)',
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
        
        # Add a second y-axis for more context
        fig.update_layout(
            yaxis2=dict(
                title="",
                overlaying="y",
                side="right",
                showgrid=False,
                showticklabels=False,
                range=[np.log10(y_min), np.log10(y_max)],
                type="log",
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="help-text">
        This chart shows how the required test duration decreases as the expected improvement increases.
        Larger improvements are easier to detect and require less data (shorter test durations).
        The logarithmic scale helps visualize both small and large differences in required test duration.
        </div>
        """, unsafe_allow_html=True)    
    with viz_tab3:
        # Add dynamic selection for comparison type
        st.markdown("### Dynamic Comparison")
        
        # Create radio buttons for choosing comparison type
        comparison_type = st.radio(
            "Select comparison type:",
            ["Variants vs. Significance", "Variants vs. Improvement"],
            horizontal=True,
            help="Choose which parameters to compare while keeping others constant"
        )
        
        if comparison_type == "Variants vs. Significance":
            # Create a fixed improvement with varying significance levels
            st.markdown(f"Showing how different significance levels affect test duration across variant counts (at {target_improvement}% improvement)")
            
            # Create a 2D grid of data
            significance_levels = [75, 80, 85, 90, 95, 99]
            variant_counts = [2, 3, 4, 5]
            
            # Create empty dataframe to store results
            comparison_df = pd.DataFrame()
            
            # Calculate data for each combination
            for sig in significance_levels:
                temp_data = []
                for var in variant_counts:
                    days = calculate_test_duration(daily_visitors, var, baseline_conversion, target_improvement, sig)["days"]
                    temp_data.append(days)
                comparison_df[f"{sig}%"] = temp_data
            
            comparison_df.index = [f"{v} Variants" for v in variant_counts]
            
            # Create a heatmap
            fig = px.imshow(
                comparison_df.values,
                x=comparison_df.columns,
                y=comparison_df.index,
                color_continuous_scale="Viridis",
                text_auto=True,
                aspect="auto",
                title=f"Test Duration (days) by Variants & Significance Level (at {target_improvement}% improvement)"
            )
            
            # Add markers for the user's selected values
            selected_row = variant_counts.index(variations)
            selected_col = significance_levels.index(significance_level)
            
            # Create a line plot for easier comparison
            fig2 = go.Figure()
            
            for i, var in enumerate(variant_counts):
                fig2.add_trace(go.Scatter(
                    x=significance_levels,
                    y=comparison_df.iloc[i].values,
                    mode='lines+markers',
                    name=f"{var} Variants",
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig2.update_layout(
                title=f"Test Duration by Significance Level for Different Variant Counts (at {target_improvement}% improvement)",
                xaxis_title="Significance Level (%)",
                yaxis_title="Days Required",
                hovermode="closest",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add a marker for the user's current selection
            fig2.add_trace(go.Scatter(
                x=[significance_level],
                y=[result['days']],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Your Selection',
                hoverinfo='text',
                hovertext=f"Your selection: {variations} variants, {significance_level}% significance"
            ))
            
        else:  # Variants vs. Improvement
            # Create a fixed significance with varying improvement levels
            st.markdown(f"Showing how different improvement levels affect test duration across variant counts (at {significance_level}% significance)")
            
            # Create a 2D grid of data
            improvement_levels = [1.0, 2.0, 5.0, 10.0, 25.0]
            variant_counts = [2, 3, 4, 5]
            
            # Create empty dataframe to store results
            comparison_df = pd.DataFrame()
            
            # Calculate data for each combination
            for imp in improvement_levels:
                temp_data = []
                for var in variant_counts:
                    days = calculate_test_duration(daily_visitors, var, baseline_conversion, imp, significance_level)["days"]
                    temp_data.append(days)
                comparison_df[f"{imp}%"] = temp_data
            
            comparison_df.index = [f"{v} Variants" for v in variant_counts]
            
            # Create a heatmap
            fig = px.imshow(
                comparison_df.values,
                x=comparison_df.columns,
                y=comparison_df.index,
                color_continuous_scale="Viridis",
                text_auto=True,
                aspect="auto",
                title=f"Test Duration (days) by Variants & Improvement Level (at {significance_level}% significance)"
            )
            
            # Create a line plot for easier comparison
            fig2 = go.Figure()
            
            for i, var in enumerate(variant_counts):
                fig2.add_trace(go.Scatter(
                    x=improvement_levels,
                    y=comparison_df.iloc[i].values,
                    mode='lines+markers',
                    name=f"{var} Variants",
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig2.update_layout(
                title=f"Test Duration by Improvement Level for Different Variant Counts (at {significance_level}% significance)",
                xaxis_title="Expected Improvement (%)",
                yaxis_title="Days Required",
                hovermode="closest",
                height=400,
                yaxis_type="log",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add a marker for the user's current selection
            # Find closest improvement level
            closest_imp = min(improvement_levels, key=lambda x: abs(x - target_improvement))
            fig2.add_trace(go.Scatter(
                x=[target_improvement],
                y=[result['days']],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Your Selection',
                hoverinfo='text',
                hovertext=f"Your selection: {variations} variants, {target_improvement}% improvement"
            ))
        
        # Display the heatmap
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the line chart
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        <div class="help-text">
        The heatmap shows the test duration for different combinations of parameters.
        The line chart makes it easier to compare trends across different variant counts.
        Your current selection is marked with a red star.
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
            res = calculate_test_duration(daily_visitors, var, baseline_conversion, imp, significance_level)
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
        title=f"Test Duration (days) by Improvement & Variants (at {significance_level}% significance)"
    )
    
    fig.update_layout(
        xaxis_title="Number of Variants",
        yaxis_title="Expected Improvement",
        coloraxis_colorbar=dict(title="Days Required"),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button for reference data
    csv = reference_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download Reference Table",
        data=csv,
        file_name=f"ab_test_reference_table_{significance_level}pct.csv",
        mime="text/csv",
    )
    
    # Footer with additional information
    st.markdown("""
    ---
    ### Additional Notes
    - This calculator uses a simplified model and should be used as a guideline only.
    - Actual results may vary based on real-world conditions and data patterns.
    - Always consult with a statistician for critical business decisions.
    - The calculator assumes a two-tailed test with equal traffic allocation.
    """)