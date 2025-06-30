import streamlit as st
import ml_stock_predictor
from dotenv import load_dotenv
load_dotenv()

from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Direction Predictor", layout="wide")
st.title("📊 Stock Prediction Summary")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Model Settings")
symbol = st.sidebar.text_input("Stock Symbol", value="SPY").upper()
window_size = st.sidebar.slider("Rolling Window Size (days)", 5, 30, value=10)
horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 20, value=5)

api_choice = "Tradier"
api_key = st.sidebar.text_input("Enter Your Tradier API Key", type="password")

st.sidebar.markdown("""
⚠️ **Disclaimer**: This model may not perform well during earnings periods or major news events. Please check for upcoming announcements before relying on predictions.
""")

if st.sidebar.button("Run Prediction"):
    if api_key.strip() == "":
        st.warning("Please provide a valid Tradier API key to continue.")
        st.stop()

    with st.spinner("Training model and making predictions..."):
        try:
            result_tuple = ml_stock_predictor.train_and_predict(symbol, window_size, horizon, api_choice, api_key)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()

        if result_tuple is None or len(result_tuple) != 16:
            st.error("Unexpected number of return values from train_and_predict().")
            st.stop()

        best_model, _, df, X_test, y_class_test, y_class_pred, y_reg_test, y_reg_pred, model_scores, conflict_flags, close_prices_pred, close_prices_test, y_class_probs, lower_bounds, upper_bounds, y_magnitude_pred = result_tuple

        if df is None or df.empty:
            st.error("No data returned from the API. Please check your symbol and API key.")
            st.stop()

        # --- Prediction Summary ---
        last_idx = -1
        if isinstance(y_class_probs, np.ndarray) and y_class_probs.ndim == 1:
            probs = [1 - y_class_probs[last_idx], y_class_probs[last_idx]]
        elif isinstance(y_class_probs, np.ndarray) and y_class_probs.ndim == 2:
            probs = y_class_probs[last_idx]
        else:
            st.error("Unexpected shape of y_class_probs")
            st.stop()

        predicted_class = np.argmax(probs)
        predicted_return = y_reg_pred[last_idx]
        last_price = df['close'].iloc[-1]
        predicted_pct_return = (np.exp(predicted_return) - 1) * 100
        predicted_direction = "⬆️ UP" if predicted_class == 1 else "⬇️ DOWN"

        lower_pct = (np.exp(lower_bounds[last_idx]) - 1) * 100
        upper_pct = (np.exp(upper_bounds[last_idx]) - 1) * 100

        # Check for contradiction between class label and probability
        if (predicted_class == 1 and probs[1] < 0.5) or (predicted_class == 0 and probs[0] < 0.5):
            st.warning("⚠️ Classification label and probability appear to conflict. Model confidence may be low.")

        st.markdown(f"""
            <div style='padding: 16px; background-color: #f4f8fb; border-left: 6px solid #0066cc; margin-bottom: 16px;'>
                <div style='font-size: 22px; font-weight: bold;'>📈 The model predicts the stock will go: <span style='color:#0066cc'>{predicted_direction}</span> over the next {horizon} day(s).</div>
                <div style='font-size: 18px; margin-top: 6px;'>Estimated net return: <strong>{predicted_pct_return:.2f}%</strong></div>
                <div style='font-size: 16px;'>⚠️ Based on <strong>log return</strong> forecast — a common method in finance to account for compounding effects. Predictions in log space help stabilize variance and reduce outlier influence, but they can also exaggerate returns if overfit. Clipping and quantile regression are used to mitigate this.</div>
                <div style='font-size: 18px;'>Expected Return Range (90% confidence, via Quantile Regression): <strong>{lower_pct:.2f}% to {upper_pct:.2f}%</strong></div>
                <div style='font-size: 18px;'>Volatility Bucket: <strong>{y_magnitude_pred[last_idx]}</strong> — Buckets represent levels of expected volatility.</div>
                <div style='font-size: 14px; margin-top: 10px;'>
                    <ul style='margin-left: 20px;'>
                        <li><strong>0</strong>: Low volatility — small returns expected.</li>
                        <li><strong>1</strong>: Medium volatility — moderate swings possible.</li>
                        <li><strong>2</strong>: High volatility — large returns are likely.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- Probabilities Table ---
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame([probs], columns=["⬇️ DOWN", "⬆️ UP"])
        st.dataframe(prob_df.style.hide(axis='index').highlight_max(axis=1, color="lightgreen"))
        st.markdown("""
        **How to Read This:**
        - `⬇️ DOWN`: Probability the stock will decline.
        - `⬆️ UP`: Probability the stock will rise.

        The higher value shows the model’s prediction confidence.
        """)

        # --- Model Accuracy ---
        st.markdown("---")
        st.subheader("Model Accuracy Overview")
        accuracy = accuracy_score(y_class_test, y_class_pred)
        st.markdown(f"**Prediction Accuracy:** {accuracy:.2%}")
        st.markdown("""
        **What This Means:**
        - This metric only measures **direction**, not how much the stock moves.
        - If accuracy = 75%, the model got the direction right 3 out of 4 times.
        - Classification is the primary model goal — regression is used to support magnitude estimation.
        """)

        # --- Optional Diagnostics Toggle ---
        with st.expander("🔍 Additional Model Diagnostics"):
            st.markdown("**Expected Return Stats (from Regression):**")
            close_prices_pred_arr = np.array(close_prices_pred)
            y_reg_pred_arr = np.array(y_reg_pred)
            expected_moves = (np.exp(y_reg_pred_arr) - 1) * close_prices_pred_arr
            st.markdown(f"- **Mean**: {np.mean(expected_moves):.2f} USD — The average expected return value.")
            st.markdown(f"- **Max**: {np.max(expected_moves):.2f} USD — The largest predicted return.")
            st.markdown(f"- **Min**: {np.min(expected_moves):.2f} USD — The largest negative return.")

            st.markdown("**Regression Fit (Supporting Info Only):**")
            close_prices_test_arr = np.array(close_prices_test)
            y_reg_test_arr = np.array(y_reg_test)
            y_reg_test_usd = (np.exp(y_reg_test_arr) - 1) * close_prices_test_arr
            y_reg_pred_usd = expected_moves
            r2 = r2_score(y_reg_test_usd, y_reg_pred_usd)
            mae = mean_absolute_error(y_reg_test_usd, y_reg_pred_usd)
            rmse = mean_squared_error(y_reg_test_usd, y_reg_pred_usd, squared=False)
            st.markdown(f"- **R² Score**: {r2:.3f} — Measures how well predictions explain actual returns. 1.0 = perfect fit, 0.0 = no better than average, negative = worse than guessing.")
            st.markdown(f"- **MAE**: {mae:.2f} USD — Typical dollar error between predicted and actual return values.")
            st.markdown(f"- **RMSE**: {rmse:.2f} USD — Like MAE but more sensitive to large errors. Useful for spotting big misses.")

            st.markdown("**Residuals Plot:** Predicted - Actual (USD)")
            residuals = y_reg_pred_usd - y_reg_test_usd
            fig, ax = plt.subplots()
            ax.hist(residuals, bins=30, color="skyblue", edgecolor="black")
            ax.axvline(0, color="red", linestyle="--")
            ax.set_xlabel("Residual (Prediction - Actual)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        st.caption(f"Debug Info: Symbol={symbol}, Window={window_size}, Horizon={horizon}, API={api_choice}")
        st.caption("Note: This dashboard emphasizes classification (direction) while using regression and quantile regression for supporting insights like return magnitude and range. Extreme values are clipped to prevent outlier distortion. Consider avoiding trades near earnings or major news events.")
