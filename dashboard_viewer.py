import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="MC Dropout Analysis", layout="wide")


DATA_PATH = "pickled/mc_dropout_epoch100/dashboard_data.pkl"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"File not found: {DATA_PATH}. Run the training script first!")
        return None
    try:
        with open(DATA_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title("MC Dropout vs Standard Dropout: Uncertainty Estimation")

    data = load_data()

    if data is None or len(data) == 0:
        st.warning("No data available. Please run the training script first to generate dashboard_data.pkl")
        st.stop()
        return

    df_list = []
    for entry in data:
        temp_df = pd.DataFrame({
            "Uncertainty (Entropy)": entry["uncertainties"],
            "Confidence": entry["confidences"],
            "Correct": entry["is_correct"],
            "Prediction": entry["predictions"],
            "Ground Truth": entry["ground_truth"]
        })
        temp_df["Method"] = entry["method"]
        temp_df["Dataset"] = entry["dataset"]
        temp_df["p"] = entry.get("p", 0.0)
        temp_df["is_dropout"] = entry.get("is_dropout", True)
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    df_no_dropout = df[df["is_dropout"] == False]
    df_dropout = df[df["is_dropout"] == True]

    # Create tabs for no-dropout and dropout models
    tab1, tab2, tab3 = st.tabs(["No Dropout Baseline (p=0.0)", "Dropout Models (p>0)", "Training Curves"])

    # tab1
    with tab1:
        st.header("No Dropout Baseline Analysis (p=0.0)")

        if len(df_no_dropout) == 0:
            st.warning("No data available for no-dropout baseline")
        else:
            st.markdown("**Model:** No dropout regularization (equivalent to p=0.0)")
            st.markdown("This is the baseline model without any dropout. It only supports standard inference.")
            # st.info("**Note:** This is the same as having dropout probability p=0.0. We train this separately as a baseline for comparison.")

            # OOD Detection for no-dropout
            st.subheader("1. Out-of-Distribution Detection")
            st.markdown("Does the baseline model know when it's seeing something new? </br>")
            st.markdown("We test this by comparing the model's uncertainty on in-distribution (CIFAR-10) vs out-of-distribution (SVHN) data.")

            ood_fig_no = px.histogram(
                df_no_dropout,
                x="Uncertainty (Entropy)",
                color="Dataset",
                barmode="overlay",
                nbins=50,
                title="Uncertainty Distribution: No Dropout Baseline",
                opacity=0.6,
                color_discrete_map={"CIFAR-10": "blue", "SVHN": "red"},
                histnorm='probability'
            )
            st.plotly_chart(ood_fig_no, use_container_width=True)

            st.info("""
**Expected behavior:**
- Without dropout, the model tends to be **overconfident**
- Blue (CIFAR-10) and Red (SVHN) histograms likely **overlap heavily**
- The model is "Confidently Wrong" on OOD data (SVHN)
- Low entropy on both in-distribution and out-of-distribution samples
""")

            st.subheader("2. Confidence Calibration")
            st.markdown("If the model says '90% confident', is it actually right 90% of the time?")

            cal_data_no = df_no_dropout[df_no_dropout["Dataset"] == "CIFAR-10"].copy()
            if len(cal_data_no) > 0:
                cal_data_no["bin"] = pd.cut(cal_data_no["Confidence"], bins=np.linspace(0, 1, 11), labels=False)

                bin_acc_no = cal_data_no.groupby("bin")["Correct"].mean()
                bin_conf_no = cal_data_no.groupby("bin")["Confidence"].mean()

                cal_fig_no = go.Figure()
                cal_fig_no.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="gray")))
                cal_fig_no.add_trace(go.Scatter(
                    x=bin_conf_no,
                    y=bin_acc_no,
                    mode="lines+markers",
                    name="No Dropout",
                    marker=dict(size=10, color="blue")
                ))
                cal_fig_no.update_layout(
                    title="Reliability Diagram (No Dropout)",
                    xaxis_title="Average Confidence",
                    yaxis_title="Accuracy (Fraction Correct)",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(cal_fig_no, use_container_width=True)

            st.subheader("3. Failure Analysis")
            failures_no = df_no_dropout[(df_no_dropout["Dataset"] == "CIFAR-10") & (df_no_dropout["Correct"] == False)]

            if len(failures_no) > 0:
                fail_type_no = st.radio("Show me failures where the model was...",
                                       ["Confident (Low Entropy)", "Uncertain (High Entropy)"],
                                       key="fail_no")

                if fail_type_no == "Confident (Low Entropy)":
                    top_failures_no = failures_no.sort_values(by="Uncertainty (Entropy)", ascending=True).head(5)
                    st.warning("These are **Silent Failures**. The model was wrong but 'sure' of itself.")
                else:
                    top_failures_no = failures_no.sort_values(by="Uncertainty (Entropy)", ascending=False).head(5)
                    st.success("These are **Good Failures**. The model was wrong, but it 'knew' it was confused.")

                st.table(top_failures_no[["Prediction", "Ground Truth", "Confidence", "Uncertainty (Entropy)"]])
            else:
                st.info("No failures found (perfect accuracy!)")

    with tab2:
        st.header("Dropout Models Analysis (p > 0)")

        if len(df_dropout) == 0:
            st.warning("No data available for dropout models")
        else:

            st.sidebar.header("Dropout Model Controls")

            unique_p_values = sorted(df_dropout["p"].unique())

            selected_p = st.sidebar.selectbox("Dropout Probability (p)", unique_p_values)

            selected_method = st.sidebar.radio("Inference Method", ["Standard", "MC Dropout"])

            subset = df_dropout[(df_dropout["Method"] == selected_method) & (df_dropout["p"] == selected_p)]

            print(df_dropout.head(10))

            st.markdown(f"**Configuration:** p={selected_p} (dropout enabled)")
            st.markdown(f"**Method:** {selected_method}")

     
            st.subheader("1. Out-of-Distribution (OOD) Detection")
            st.markdown("Does the model know when it's seeing something new (SVHN)?")

            # CIFAR-10 (ID) vs SVHN (OOD) for the selected method
            ood_fig = px.histogram(
                subset,
                x="Uncertainty (Entropy)",
                color="Dataset",
                barmode="overlay",
                nbins=50,
                title=f"Uncertainty Distribution: {selected_method} (p={selected_p})",
                opacity=0.6,
                color_discrete_map={"CIFAR-10": "blue", "SVHN": "red"},
                histnorm='probability'
            )
            st.plotly_chart(ood_fig, use_container_width=True)

            st.info("""
**Expected behavior:**
- **Standard:** The Blue and Red histograms will likely overlap heavily. The model is "Confidently Wrong" on SVHN.
- **MC Dropout:** The Red (SVHN) histogram should shift to the **right** (Higher Entropy). The model successfully flags SVHN as "Unknown."
- **Higher dropout rates (p):** Generally lead to higher uncertainty, especially with MC Dropout.
""")

            st.subheader("2. Confidence Calibration")
            st.markdown("If the model says '90% confident', is it actually right 90% of the time?")

            # Expected Calibration Error (ECE) 
           
            cal_data = subset[subset["Dataset"] == "CIFAR-10"].copy() # only calc on ID data
            if len(cal_data) > 0:
                cal_data["bin"] = pd.cut(cal_data["Confidence"], bins=np.linspace(0, 1, 11), labels=False)

                bin_acc = cal_data.groupby("bin")["Correct"].mean()
                bin_conf = cal_data.groupby("bin")["Confidence"].mean()

                cal_fig = go.Figure()

                cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="gray")))

                cal_fig.add_trace(go.Scatter(
                    x=bin_conf,
                    y=bin_acc,
                    mode="lines+markers",
                    name=f"{selected_method}",
                    marker=dict(size=10, color="orange")
                ))

                cal_fig.update_layout(
                    title=f"Reliability Diagram ({selected_method}, p={selected_p})",
                    xaxis_title="Average Confidence",
                    yaxis_title="Accuracy (Fraction Correct)",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1])
                )

                st.plotly_chart(cal_fig, use_container_width=True)


            st.subheader("3. Failure Gallery")
            st.markdown("Inspect specific cases where the model failed.")

            failures = subset[(subset["Dataset"] == "CIFAR-10") & (subset["Correct"] == False)]

            if len(failures) > 0:
                fail_type = st.radio("Show me failures where the model was...",
                                   ["Confident (Low Entropy)", "Uncertain (High Entropy)"],
                                   key="fail_dropout")

                if fail_type == "Confident (Low Entropy)":
                    top_failures = failures.sort_values(by="Uncertainty (Entropy)", ascending=True).head(5)
                    st.warning("These are **Silent Failures**. The model was wrong but 'sure' of itself.")
                else:
                    top_failures = failures.sort_values(by="Uncertainty (Entropy)", ascending=False).head(5)
                    st.success("These are **Good Failures**. The model was wrong, but it 'knew' it was confused.")

                st.table(top_failures[["Prediction", "Ground Truth", "Confidence", "Uncertainty (Entropy)"]])
            else:
                st.info("No failures found (perfect accuracy!)")


            st.subheader("4. Dropout Configuration Comparison")
            st.markdown("Compare performance across different dropout probabilities")

            summary_data = []
            for p_val in unique_p_values:
                for method in ["Standard", "MC Dropout"]:
                    config_subset = df_dropout[(df_dropout["p"] == p_val) & (df_dropout["Method"] == method) & (df_dropout["Dataset"] == "CIFAR-10")]
                    if len(config_subset) > 0:
                        summary_data.append({
                            "p": p_val,
                            "Method": method,
                            "Accuracy": config_subset["Correct"].mean(),
                            "Avg Confidence": config_subset["Confidence"].mean(),
                            "Avg Uncertainty": config_subset["Uncertainty (Entropy)"].mean()
                        })

            if len(summary_data) > 0:
                summary_df = pd.DataFrame(summary_data)

                col1, col2 = st.columns(2)

                with col1:
                    acc_fig = px.bar(
                        summary_df,
                        x="p",
                        y="Accuracy",
                        color="Method",
                        barmode="group",
                        title="Accuracy by Dropout Probability",
                        color_discrete_map={"Standard": "lightblue", "MC Dropout": "darkblue"}
                    )
                    st.plotly_chart(acc_fig, use_container_width=True)

                with col2:
                    unc_fig = px.bar(
                        summary_df,
                        x="p",
                        y="Avg Uncertainty",
                        color="Method",
                        barmode="group",
                        title="Average Uncertainty by Dropout Probability",
                        color_discrete_map={"Standard": "lightcoral", "MC Dropout": "darkred"}
                    )
                    st.plotly_chart(unc_fig, use_container_width=True)

                st.subheader("Summary Statistics")
                st.dataframe(summary_df.style.format({
                    "Accuracy": "{:.4f}",
                    "Avg Confidence": "{:.4f}",
                    "Avg Uncertainty": "{:.4f}"
                }))

                st.info("""
**Key Insights:**
- **MC Dropout** typically provides better calibrated uncertainty estimates
- **Higher dropout rates** (larger p) increase uncertainty but may reduce accuracy
""")


    with tab3:
        st.header("Training Curves")
        st.markdown("Visualize training and test loss/accuracy over epochs")

        try:
            try :
                with open('pickled/mc_dropout_epoch100/grid_search_results.pkl', 'rb') as f:
                    grid_results = pickle.load(f)
            except FileNotFoundError:
                st.error("File not found: pickled/mc_dropout_epoch100/grid_search_results.pkl. Run the training script first!")
                st.stop()

            no_dropout_results = [r for r in grid_results if not r['is_dropout']]
            dropout_results = [r for r in grid_results if r['is_dropout']]

            subtab1, subtab2 = st.tabs(["No Dropout (p=0.0)", "Dropout Models (p>0)"])

            with subtab1:
                if len(no_dropout_results) > 0:
                    result = no_dropout_results[0]
                    epochs = list(range(1, len(result['train_losses']) + 1))

                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=epochs, y=result['train_losses'], mode='lines+markers', name='Train Loss'))
                    fig_loss.add_trace(go.Scatter(x=epochs, y=result['test_losses'], mode='lines+markers', name='Test Loss'))
                    fig_loss.update_layout(title="No Dropout (p=0.0): Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_loss, use_container_width=True)

                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(x=epochs, y=result['accs'], mode='lines+markers', name='Test Accuracy'))
                    fig_acc.update_layout(title="No Dropout (p=0.0): Test Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                    st.plotly_chart(fig_acc, use_container_width=True)

                    st.metric("Final Test Accuracy", f"{result['final_acc']:.4f}")
                else:
                    st.warning("No training data available for no-dropout model")

            with subtab2:
                if len(dropout_results) > 0:
                    p_values = sorted([r['p'] for r in dropout_results])
                    selected_p_curve = st.selectbox("Select Dropout Probability (p > 0)", p_values, key="curve_p")

                    result = [r for r in dropout_results if r['p'] == selected_p_curve][0]
                    epochs = list(range(1, len(result['train_losses']) + 1))

                    # Loss 
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=epochs, y=result['train_losses'], mode='lines+markers', name='Train Loss'))
                    fig_loss.add_trace(go.Scatter(x=epochs, y=result['test_losses'], mode='lines+markers', name='Test Loss'))
                    fig_loss.update_layout(title=f"Dropout p={selected_p_curve}: Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_loss, use_container_width=True)

                    # Accuracy 
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(x=epochs, y=result['accs_det'], mode='lines+markers', name='Deterministic Accuracy'))
                    fig_acc.add_trace(go.Scatter(x=epochs, y=result['accs_mc'], mode='lines+markers', name='MC Dropout Accuracy'))
                    fig_acc.update_layout(title=f"Dropout p={selected_p_curve}: Test Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                    st.plotly_chart(fig_acc, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Deterministic Accuracy", f"{result['final_acc_det']:.4f}")
                    with col2:
                        st.metric("Final MC Dropout Accuracy", f"{result['final_acc_mc']:.4f}")
                    with col3:
                        improvement = (result['final_acc_mc'] - result['final_acc_det']) * 100
                        st.metric("MC Improvement", f"{improvement:.2f}%")
                else:
                    st.warning("No training data available for dropout models")

            
            st.markdown("---")
            st.header("Comparison Plots")

            comp_tab1, comp_tab2 = st.tabs(["All Dropout Configurations", "Individual dropout configurations vs No-Dropout"])

            with comp_tab1:
                st.subheader("No Dropout vs Standard Dropout Across All p Values")
                st.markdown("Compare how No Dropout, MC Dropout and Standard Dropout perform across different dropout probabilities")

                if len(dropout_results) > 0:
                    
                    epochs = list(range(1, len(dropout_results[0]['train_losses']) + 1))

                    # Test Loss Comparison
                    fig_loss_comp = go.Figure()

                    # Add test loss for no dropout case
                    fig_loss_comp.add_trace(go.Scatter(
                        x=epochs,
                        y=no_dropout_results[0]['test_losses'],
                        mode='lines',
                        name='no dropout',
                        line=dict(dash='solid')
                    ))

                    # Add Standard (deterministic) test loss for each p
                    for result in dropout_results:
                        p = result['p']
                        fig_loss_comp.add_trace(go.Scatter(
                            x=epochs,
                            y=result['test_losses'],
                            mode='lines',
                            name=f'Standard p={p}',
                            line=dict(dash='solid')
                        ))

                    fig_loss_comp.update_layout(
                        title="Test Loss: All Dropout Configurations",
                        xaxis_title="Epoch",
                        yaxis_title="Test Loss",
                        hovermode='x unified',
                        height=700
                    )
                    st.plotly_chart(fig_loss_comp, use_container_width=True)

                    # Test Accuracy Comparison - Standard vs MC
                    fig_acc_comp = go.Figure()

                    # Add accuracy for no dropout case
                    fig_acc_comp.add_trace(go.Scatter(
                        x=epochs,
                        y=no_dropout_results[0]['accs'],
                        mode='lines',
                        name='no dropout',
                        line=dict(dash='solid')
                    ))

                    # Add Standard accuracy
                    for result in dropout_results:
                        p = result['p']
                        fig_acc_comp.add_trace(go.Scatter(
                            x=epochs,
                            y=result['accs_det'],
                            mode='lines',
                            name=f'Standard p={p}',
                            line=dict(dash='dash')
                        ))

                    # Add MC Dropout accuracy
                    for result in dropout_results:
                        p = result['p']
                        fig_acc_comp.add_trace(go.Scatter(
                            x=epochs,
                            y=result['accs_mc'],
                            mode='lines',
                            name=f'MC Dropout p={p}',
                            line=dict(dash='solid')
                        ))

                    fig_acc_comp.update_layout(
                        title="Test Accuracy: Standard vs MC Dropout Across All p Values",
                        xaxis_title="Epoch",
                        yaxis_title="Accuracy",
                        hovermode='x unified',
                        height=700
                    )
                    st.plotly_chart(fig_acc_comp, use_container_width=True)

                    # Train Loss Comparison
                    fig_train_comp = go.Figure()

                    # Add train loss for no dropout case
                    fig_train_comp.add_trace(go.Scatter(
                        x=epochs,
                        y=no_dropout_results[0]['train_losses'],
                        mode='lines',
                        name='no dropout',
                        line=dict(dash='solid')
                    ))

                    for result in dropout_results:
                        p = result['p']
                        fig_train_comp.add_trace(go.Scatter(
                            x=epochs,
                            y=result['train_losses'],
                            mode='lines',
                            name=f'Train Loss p={p}'
                        ))

                    fig_train_comp.update_layout(
                        title="Training Loss: All Dropout Configurations",
                        xaxis_title="Epoch",
                        yaxis_title="Train Loss",
                        hovermode='x unified',
                        height=700
                    )
                    st.plotly_chart(fig_train_comp, use_container_width=True)

                else:
                    st.warning("No dropout results available for comparison")

            with comp_tab2:
                st.subheader("Detailed Comparison: Different dropout configurations vs No-Dropout Baseline")
                st.markdown("Compare the no-dropout baseline against different dropout configurations (Standard & MC)")

                for r in dropout_results:
                    if len(no_dropout_results) > 0:
                        no_drop = no_dropout_results[0]
                        epochs = list(range(1, len(r['train_losses']) + 1))

                        # Plot 1: Train Loss Comparison
                        fig_train = go.Figure()
                        fig_train.add_trace(go.Scatter(
                            x=epochs,
                            y=no_drop['train_losses'],
                            mode='lines',
                            name='No Dropout (p=0.0)',
                            line=dict(color='red', width=2)
                        ))
                        fig_train.add_trace(go.Scatter(
                            x=epochs,
                            y=r['train_losses'],
                            mode='lines',
                            name=f"Dropout p={r['p']}",
                            line=dict(color='blue', width=2)
                        ))
                        fig_train.update_layout(
                            title=f"Training Loss: No-Dropout vs Dropout p={r['p']}",
                            xaxis_title="Epoch",
                            yaxis_title="Train Loss",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_train, use_container_width=True)

                        # Plot 2: Test Loss Comparison
                        fig_test = go.Figure()
                        fig_test.add_trace(go.Scatter(
                            x=epochs,
                            y=no_drop['test_losses'],
                            mode='lines',
                            name='No Dropout (p=0.0)',
                            line=dict(color='red', width=2)
                        ))
                        fig_test.add_trace(go.Scatter(
                            x=epochs,
                            y=r['test_losses'],
                            mode='lines',
                            name=f"Dropout p={r['p']}",
                            line=dict(color='blue', width=2)
                        ))
                        fig_test.update_layout(
                            title=f"Test Loss: No-Dropout vs Dropout p={r['p']}",
                            xaxis_title="Epoch",
                            yaxis_title="Test Loss",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_test, use_container_width=True)

                        # Test Accuracy Comparison (3 methods)
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(
                            x=epochs,
                            y=no_drop['accs'],
                            mode='lines',
                            name='No Dropout (p=0.0)',
                            line=dict(color='red', width=2, dash='solid')
                        ))
                        fig_acc.add_trace(go.Scatter(
                            x=epochs,
                            y=r['accs_det'],
                            mode='lines',
                            name=f"Dropout p={r['p']} (Standard)",
                            line=dict(color='blue', width=2, dash='dash')
                        ))
                        fig_acc.add_trace(go.Scatter(
                            x=epochs,
                            y=r['accs_mc'],
                            mode='lines',
                            name=f"Dropout p={r['p']} (MC)",
                            line=dict(color='green', width=2, dash='solid')
                        ))
                        fig_acc.update_layout(
                            title=f"Test Accuracy: No-Dropout vs Dropout p={r['p']} (Standard & MC)",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)

                        st.markdown("### Final Metrics Comparison")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "No Dropout (p=0.0)",
                                f"{no_drop['final_acc']:.4f}",
                                help="Baseline without regularization"
                            )

                        with col2:
                            # delta_std = (p03['final_acc_det'] - no_drop['final_acc']) * 100
                            st.metric(
                                f"Dropout p={r['p']} (Standard)",
                                f"{r['final_acc_det']:.4f}",
                                # f"{delta_std:+.2f}%",
                                help="With dropout regularization"
                            )

                        with col3:
                            # delta_mc = (p03['final_acc_mc'] - no_drop['final_acc']) * 100
                            st.metric(
                                f"Dropout p={r['p']} (MC)",
                                f"{r['final_acc_mc']:.4f}",
                                # f"{delta_mc:+.2f}%",
                                help="MC Dropout averaging"
                            )

        except FileNotFoundError:
            st.error("Training results not found. Please run main.py first to generate grid_search_results.pkl")


if __name__ == "__main__":
    main()