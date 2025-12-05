"""
Interactive Streamlit Dashboard for Code Documentation Generation
Allows side-by-side comparison of different transformer architectures
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Code Documentation Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"> Transformer Code Documentation Generator</div>', 
            unsafe_allow_html=True)
st.markdown("Compare different transformer architectures for generating code documentation")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    st.subheader("Select Models to Compare")
    use_codebert = st.checkbox("CodeBERT (Encoder-only)", value=True)
    use_codellama = st.checkbox("CodeLlama (Decoder-only)", value=False)
    use_codet5 = st.checkbox("CodeT5 (Encoder-decoder)", value=True)
    
    st.divider()
    
    # Generation parameters
    st.subheader("Generation Parameters")
    max_length = st.slider("Max Length", 50, 512, 128)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    num_beams = st.slider("Num Beams", 1, 10, 4)
    
    st.divider()
    
    # Language selection
    language = st.selectbox(
        "Programming Language",
        ["python", "java", "javascript", "go", "ruby", "php"]
    )

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "Generate Documentation", 
    "Performance Metrics",
    "Model Comparison",
    "Analysis"
])

# Tab 1: Generate Documentation
with tab1:
    st.header("Generate Documentation")
    
    # Example codes
    example_codes = {
        "Fibonacci": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
        
        "Binary Search": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
        
        "Quick Sort": """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)"""
    }
    
    # Example selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_example = st.selectbox("Select Example", ["Custom"] + list(example_codes.keys()))
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
    
    # Code input
    if selected_example == "Custom":
        code_input = st.text_area(
            "Enter your code:",
            height=200,
            placeholder="Paste your code here..."
        )
    else:
        code_input = st.text_area(
            "Enter your code:",
            value=example_codes[selected_example],
            height=200
        )
    
    # Generate button
    if st.button("Generate Documentation", type="primary", use_container_width=True):
        if not code_input.strip():
            st.error("Please enter some code!")
        else:
            with st.spinner("Generating documentation..."):
                # Create columns for each selected model
                models_selected = []
                if use_codebert:
                    models_selected.append("CodeBERT")
                if use_codellama:
                    models_selected.append("CodeLlama")
                if use_codet5:
                    models_selected.append("CodeT5")
                
                if not models_selected:
                    st.warning("Please select at least one model in the sidebar!")
                else:
                    cols = st.columns(len(models_selected))
                    
                    for idx, (col, model_name) in enumerate(zip(cols, models_selected)):
                        with col:
                            st.markdown(f'<div class="model-header">{model_name}</div>', 
                                      unsafe_allow_html=True)
                            
                            # Simulate generation (replace with actual model inference)
                            import time
                            start_time = time.time()
                            
                            # Demo output
                            demo_docs = {
                                "CodeBERT": f"This function implements {selected_example if selected_example != 'Custom' else 'the algorithm'}. "
                                           f"It takes input parameters and returns the computed result using efficient methods.",
                                "CodeLlama": f"**Purpose**: Implements {selected_example if selected_example != 'Custom' else 'the algorithm'}\n\n"
                                            f"**Parameters**: Input values for computation\n\n"
                                            f"**Returns**: Processed result\n\n"
                                            f"**Complexity**: Depends on input size",
                                "CodeT5": f"Function that performs {selected_example if selected_example != 'Custom' else 'computation'}. "
                                         f"Uses standard algorithmic approach for optimal performance."
                            }
                            
                            time.sleep(0.5)  # Simulate processing
                            inference_time = time.time() - start_time
                            
                            # Display documentation
                            st.markdown("**Generated Documentation:**")
                            st.info(demo_docs.get(model_name, "Documentation generated."))
                            
                            # Metrics
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            metric_cols = st.columns(2)
                            with metric_cols[0]:
                                st.metric("Inference Time", f"{inference_time*1000:.0f} ms")
                            with metric_cols[1]:
                                st.metric("Quality Score", f"{85 + idx*3:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Performance Metrics
with tab2:
    st.header("Performance Metrics")
    
    # Sample metrics data
    metrics_data = {
        'Model': ['CodeBERT', 'CodeLlama-7B', 'CodeT5-base'],
        'BLEU': [45.2, 52.8, 58.3],
        'ROUGE-L': [48.7, 55.1, 61.4],
        'CodeBLEU': [42.3, 49.6, 55.2],
        'BERTScore': [82.1, 85.3, 87.9],
        'Inference Time (ms)': [45, 120, 68]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Display table
    st.subheader("Metrics Comparison Table")
    st.dataframe(df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for metrics
        fig = go.Figure()
        for metric in ['BLEU', 'ROUGE-L', 'CodeBLEU']:
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Model'],
                y=df[metric]
            ))
        fig.update_layout(
            title="Quality Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Inference time comparison
        fig = px.bar(
            df,
            x='Model',
            y='Inference Time (ms)',
            title="Inference Time Comparison",
            color='Inference Time (ms)',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Model Comparison
with tab3:
    st.header("Architecture Comparison")
    
    # Model details
    model_details = {
        "CodeBERT": {
            "Architecture": "Encoder-only (BERT-based)",
            "Parameters": "125M",
            "Pre-training": "Code-text pairs",
            "Strengths": "Understanding code structure, fast inference",
            "Weaknesses": "Limited generation capability",
            "Best For": "Short documentation, code understanding"
        },
        "CodeLlama": {
            "Architecture": "Decoder-only (GPT-based)",
            "Parameters": "7B",
            "Pre-training": "Code repositories",
            "Strengths": "Flexible generation, follows prompts well",
            "Weaknesses": "Slower inference, larger memory",
            "Best For": "Detailed documentation, multiple languages"
        },
        "CodeT5": {
            "Architecture": "Encoder-decoder (T5-based)",
            "Parameters": "220M",
            "Pre-training": "Code translation tasks",
            "Strengths": "Balanced performance, seq2seq design",
            "Weaknesses": "Moderate speed",
            "Best For": "General documentation, balanced quality/speed"
        }
    }
    
    # Display in columns
    cols = st.columns(3)
    for idx, (model_name, details) in enumerate(model_details.items()):
        with cols[idx]:
            st.markdown(f'<div class="model-header">{model_name}</div>', 
                       unsafe_allow_html=True)
            for key, value in details.items():
                st.markdown(f"**{key}:** {value}")
            st.divider()

# Tab 4: Analysis
with tab4:
    st.header("Performance Analysis")
    
    # Trade-off analysis
    st.subheader("Quality vs. Speed Trade-off")
    
    trade_off_data = pd.DataFrame({
        'Model': ['CodeBERT', 'CodeLlama-7B', 'CodeT5-base'],
        'Quality Score': [45.2, 58.3, 52.8],
        'Inference Time (ms)': [45, 120, 68],
        'Parameters (M)': [125, 7000, 220]
    })
    
    fig = px.scatter(
        trade_off_data,
        x='Inference Time (ms)',
        y='Quality Score',
        size='Parameters (M)',
        color='Model',
        title="Quality vs. Speed Trade-off",
        labels={'Quality Score': 'Average Quality Score (%)'},
        hover_data=['Parameters (M)']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Recommendations")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.success("**For Speed:** Use CodeBERT")
        st.write("Best when inference time is critical and documentation needs are simple.")
    
    with rec_col2:
        st.info("**For Quality:** Use CodeLlama")
        st.write("Best when documentation quality is paramount and resources are available.")
    
    with rec_col3:
        st.warning("**For Balance:** Use CodeT5")
        st.write("Best general-purpose option balancing quality and speed.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Transformer Code Documentation Generator</strong></p>
    <p>DS5760 - GenAI Project | Vanderbilt University</p>
</div>
""", unsafe_allow_html=True)
