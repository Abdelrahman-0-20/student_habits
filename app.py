import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
import warnings
warnings.filterwarnings('ignore')

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Student Habits & Performance", layout="wide")
st.title(" Student Habits vs Academic Performance")
st.markdown("Explore, filter, and predict performance using only the most predictive habits.")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("student_habits_performance.csv")
    return df

df = load_data()

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "EDA & Visualization",
    "Filter & Download Data",
    "Case Study",
    "Machine Learning"
])

# ============================================================
# 1. EDA & VISUALIZATION
# ============================================================
if section == "EDA & Visualization":
    st.header(" Exploratory Data Analysis")
    with st.expander("Show raw data"):
        st.dataframe(df.head(10))
        st.caption(f"Shape: {df.shape}")

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include='all'))

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values!")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Distribution plot
    st.subheader("Distribution of Numeric Features")
    col1, col2 = st.columns(2)
    with col1:
        num_feature = st.selectbox("Select numeric column", numeric_cols)
    with col2:
        bin_count = st.slider("Number of bins", 5, 50, 20)
    fig = px.histogram(df, x=num_feature, nbins=bin_count, marginal="box",
                       title=f"Distribution of {num_feature}")
    st.plotly_chart(fig, use_container_width=True)

    if categorical_cols:
        st.subheader("Categorical Feature Counts")
        cat_feature = st.selectbox("Select categorical column", categorical_cols)
        fig = px.histogram(df, x=cat_feature, color=cat_feature,
                           title=f"Count of {cat_feature}")
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) > 1:
        st.subheader("Correlation Heatmap (Numeric Features)")
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Generate Pairplot (may be slow)"):
        sample_df = df[numeric_cols].sample(min(200, len(df)), random_state=42)
        # Add a target if exists
        target_candidates = ['Performance', 'Grade', 'Performance_Category', 'Result']
        target = None
        for col in target_candidates:
            if col in df.columns:
                target = col
                break
        if target and target in categorical_cols:
            sample_df[target] = df.loc[sample_df.index, target]
            sns.pairplot(sample_df, hue=target, diag_kind='kde')
        else:
            sns.pairplot(sample_df)
        st.pyplot(plt.gcf())
        plt.clf()

# ============================================================
# 2. FILTER & DOWNLOAD DATA
# ============================================================
elif section == "Filter & Download Data":
    st.header(" Filter and Download Data")
    filtered_df = df.copy()
    st.sidebar.subheader("Filters")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            selected_range = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]
        elif df[col].nunique() < 20:
            options = df[col].unique().tolist()
            selected = st.sidebar.multiselect(f"{col}", options, default=options)
            filtered_df = filtered_df[filtered_df[col].isin(selected)]
        else:
            search_term = st.sidebar.text_input(f"Search in {col}")
            if search_term:
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(search_term, case=False)]
    st.write(f"Filtered rows: {len(filtered_df)} / {len(df)}")
    st.dataframe(filtered_df.head(20))
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(" Download filtered data as CSV", csv, "filtered_student_data.csv", "text/csv")

# ============================================================
# 3. CASE STUDY (improved)
# ============================================================
elif section == "Case Study":
    st.header(" Case Study: How Habits Affect Performance")

    # Step 1: Let user choose the performance indicator
    possible_targets = [col for col in df.columns if df[col].nunique() < 20]  # low cardinality columns likely categories
    # Also include numeric columns that might be binned
    all_cols = df.columns.tolist()
    target = st.selectbox("Choose the performance indicator", options=all_cols,
                         index=all_cols.index('Performance') if 'Performance' in all_cols else 0)

    # Step 2: Handle numeric vs categorical target
    if df[target].dtype == 'object' or df[target].nunique() < 10:
        cat_target = df[target].astype(str)
        st.success(f"Using '{target}' as categorical performance indicator.")
    else:
        # Numeric target: bin into quartiles for grouping
        st.info(f"'{target}' is numeric – it will be grouped into quartiles for analysis.")
        df['performance_group'] = pd.qcut(df[target], q=4, labels=['Low', 'Below Avg', 'Above Avg', 'High'])
        cat_target = df['performance_group']
        target = 'performance_group'  # update to use the new binned column

    # Step 3: Show average numeric habits per group
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Remove the target itself if numeric (already grouped)
    if target in numeric_cols:
        numeric_cols.remove(target)
    if 'performance_group' in numeric_cols:
        numeric_cols.remove('performance_group')

    st.subheader(f"Average Habits by {target}")
    if numeric_cols:
        grouped = df.groupby(target)[numeric_cols].mean().round(2)
        st.dataframe(grouped)

        selected_habit = st.selectbox("Select habit to compare", numeric_cols)
        fig = px.bar(grouped, x=grouped.index, y=selected_habit,
                     title=f"Average {selected_habit} per {target}",
                     labels={'x': target, 'y': selected_habit})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No numeric habits to compare.")

    # Step 4: Correlation with encoded target
    st.subheader("Correlation with Performance (Numeric Encoded)")
    le = LabelEncoder()
    encoded_target = le.fit_transform(df[target].astype(str))
    # Compute correlations for numeric columns only
    if numeric_cols:
        corrs = {}
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'float64']:
                corrs[col] = np.corrcoef(df[col].fillna(df[col].median()), encoded_target)[0, 1]
        corr_df = pd.DataFrame.from_dict(corrs, orient='index', columns=['Correlation']).sort_values('Correlation', ascending=False)
        fig = px.bar(corr_df, orientation='h', labels={'index': 'Feature', 'value': 'Correlation'},
                     title="Linear Correlation with Performance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No numeric features to correlate.")

    # Step 5: Dynamic insights based on top correlations
    if numeric_cols:
        top_pos = corr_df.head(3).index.tolist() if len(corr_df) >= 3 else corr_df.index.tolist()
        top_neg = corr_df.tail(3).index.tolist() if len(corr_df) >= 3 else []
        st.subheader("Key Insights")
        st.markdown(f"""
        - **Top positive factors** associated with better performance: {', '.join(top_pos)}.
        - **Top negative factors**: {', '.join(top_neg) if top_neg else 'None strongly negative'}.
        - Students with higher values in positive factors tend to perform better.
        """)

# ============================================================
# 4. MACHINE LEARNING (only predictive features)
# ============================================================
elif section == "Machine Learning":
    st.header(" Predict Student Performance (Using Only Predictive Features)")

    # Choose target
    all_cols = df.columns.tolist()
    target = st.selectbox("Select target column (categorical or numeric)", all_cols,
                          index=all_cols.index('Performance') if 'Performance' in all_cols else 0)

    # Determine problem type
    if df[target].dtype == 'object' or df[target].nunique() < 10:
        problem_type = "Classification"
        st.info(f"Classification with {df[target].nunique()} classes.")
    else:
        problem_type = "Regression"
        st.info("Regression problem (numeric target).")

    # Exclude target from features
    feature_candidates = [c for c in all_cols if c != target]

    if not feature_candidates:
        st.error("No feature columns available.")
    else:
        # ------------------ FEATURE SELECTION ------------------
        st.subheader(" Automatic Feature Selection (keep only predictive ones)")
        st.markdown("We'll score each feature's predictive power and let you choose a threshold.")

        # Prepare data for scoring
        X_full = df[feature_candidates].copy()
        y_full = df[target].copy()

        # Encode categorical features for selection
        cat_features = X_full.select_dtypes(include='object').columns.tolist()
        if cat_features:
            X_encoded = pd.get_dummies(X_full, columns=cat_features, drop_first=True)
        else:
            X_encoded = X_full.copy()

        # Encode target if classification
        if problem_type == "Classification":
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y_full.astype(str))
        else:
            y_encoded = y_full.astype(float)

        # Handle missing values (median for numeric, mode for categorical encoded)
        for col in X_encoded.columns:
            if X_encoded[col].isnull().any():
                if X_encoded[col].dtype == 'object':
                    X_encoded[col].fillna(X_encoded[col].mode()[0], inplace=True)
                else:
                    X_encoded[col].fillna(X_encoded[col].median(), inplace=True)

        # Compute feature importance using Random Forest (or mutual info)
        if st.button("Compute Feature Importance Scores"):
            with st.spinner("Training a quick model to rank features..."):
                if problem_type == "Classification":
                    selector = SelectKBest(score_func=mutual_info_classif, k='all')
                else:
                    selector = SelectKBest(score_func=mutual_info_regression, k='all')
                selector.fit(X_encoded, y_encoded)
                scores = selector.scores_
                feature_scores = pd.DataFrame({
                    'feature': X_encoded.columns,
                    'importance': scores
                }).sort_values('importance', ascending=False)

                st.session_state['feature_scores'] = feature_scores
                st.success("Feature scores computed!")

        if 'feature_scores' in st.session_state:
            feature_scores = st.session_state['feature_scores']
            # Show bar chart
            fig = px.bar(feature_scores.head(20), x='importance', y='feature', orientation='h',
                         title="Top 20 Feature Importance Scores (Mutual Information)")
            st.plotly_chart(fig, use_container_width=True)

            # Threshold slider
            max_score = feature_scores['importance'].max()
            min_score = feature_scores['importance'].min()
            threshold = st.slider("Importance threshold (features with score above this are kept)",
                                  min_value=float(min_score), max_value=float(max_score),
                                  value=float(np.percentile(feature_scores['importance'], 50)))

            selected_features = feature_scores[feature_scores['importance'] >= threshold]['feature'].tolist()
            st.write(f"**{len(selected_features)} features selected** out of {len(feature_scores)}")

            # Map back to original column names (some may be dummy columns)
            # We'll just use the encoded column names for training; they correspond to original features
            st.markdown("**Selected features (encoded names):**")
            st.write(selected_features)

            # Train final model on selected features
            st.subheader(" Train Final Model on Selected Features")
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
            model_choice = st.selectbox("Choose final model",
                                        ["Random Forest", "Logistic Regression"] if problem_type == "Classification" else ["Random Forest", "Linear Regression"])

            if st.button("Train Final Model"):
                X_selected = X_encoded[selected_features]
                y = y_encoded

                # Split
                X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)

                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Model
                if problem_type == "Classification":
                    if model_choice == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    else:
                        n_estimators = st.slider("Number of trees", 50, 300, 100)
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                else:
                    if model_choice == "Linear Regression":
                        model = LinearRegression()
                    else:
                        n_estimators = st.slider("Number of trees", 50, 300, 100)
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                # Metrics
                if problem_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Accuracy: {acc:.2f}")
                    st.subheader("Classification Report")
                    if le_target:
                        target_names = le_target.classes_
                    else:
                        target_names = [str(x) for x in np.unique(y)]
                    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, text_auto=True, x=target_names, y=target_names,
                                    labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    st.success(f"R² Score: {r2:.2f}  |  RMSE: {rmse:.2f}")

                # Feature importance (if tree-based)
                if "Random Forest" in model_choice:
                    importances = model.feature_importances_
                    feat_imp = pd.DataFrame({'feature': selected_features, 'importance': importances}).sort_values('importance', ascending=False)
                    fig = px.bar(feat_imp, x='importance', y='feature', orientation='h',
                                 title="Final Model Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)

                # Save for prediction
                st.session_state['final_model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['selected_features'] = selected_features
                st.session_state['cat_features'] = cat_features
                st.session_state['feature_candidates'] = feature_candidates
                if problem_type == "Classification":
                    st.session_state['le_target'] = le_target
                st.session_state['problem_type'] = problem_type

        # Live prediction form
        if 'final_model' in st.session_state:
            st.markdown("---")
            st.subheader(" Live Prediction (Only Predictive Features)")
            st.write("Enter values for the selected features:")

            input_data = {}
            cols = st.columns(2)
            # Map selected encoded features back to original column name requests
            # We need original columns that produced these dummy columns
            # For simplicity, we'll prompt for original columns that are part of the encoded names
            original_needed = set()
            for feat in st.session_state['selected_features']:
                # Check if feat contains '_' which might be a dummy; get base column
                parts = feat.split('_')
                # This is tricky; we'll just request all original feature columns that appear in the selected encoded set
                # Better: we can derive required inputs from original columns by seeing if any dummy starts with col_
                found = False
                for orig_col in st.session_state['feature_candidates']:
                    if orig_col in st.session_state['cat_features']:
                        # categorical: check if any selected feature starts with orig_col + '_'
                        if any(f.startswith(orig_col + '_') for f in st.session_state['selected_features']):
                            original_needed.add(orig_col)
                            found = True
                            break
                    else:
                        # numeric: check if exact name in selected_features
                        if orig_col in st.session_state['selected_features']:
                            original_needed.add(orig_col)
                            found = True
                            break
                if not found:
                    # Could be an encoded name not directly linked (e.g., 'feature_value')
                    # Fallback: use the encoded name itself as a numeric input
                    original_needed.add(feat)

            # Now show input fields for original_needed
            for i, col in enumerate(sorted(original_needed)):
                with cols[i % 2]:
                    if col in df.columns:
                        if df[col].dtype == 'object':
                            unique_vals = df[col].dropna().unique().tolist()
                            input_data[col] = st.selectbox(f"{col}", unique_vals)
                        else:
                            min_val = float(df[col].min())
                            max_val = float(df[col].max())
                            mean_val = float(df[col].mean())
                            input_data[col] = st.number_input(f"{col}", min_val, max_val, value=mean_val)
                    else:
                        # It's an encoded name (dummy) not in original df; we'll still accept a numeric input
                        input_data[col] = st.number_input(f"{col} (encoded)", value=0.0)

            if st.button("Predict"):
                # Build input DataFrame
                input_df = pd.DataFrame([input_data])
                # Encode categoricals present in input_df
                cat_cols_in_input = [c for c in input_df.columns if c in st.session_state['cat_features']]
                if cat_cols_in_input:
                    # We need the same dummy encoding as training. Use pd.get_dummies with same categories?
                    # Safer: reuse a fitted encoder, but we don't have one. Simpler: we'll just use the same dummy columns as training.
                    # We'll manually create dummy columns and align with selected_features.
                    input_encoded = pd.get_dummies(input_df, columns=cat_cols_in_input, drop_first=True)
                else:
                    input_encoded = input_df.copy()

                # Add missing dummy columns that might be in selected_features but not in input
                for feat in st.session_state['selected_features']:
                    if feat not in input_encoded.columns:
                        input_encoded[feat] = 0
                # Keep only the selected features and order them
                input_final = input_encoded[st.session_state['selected_features']]
                # Scale
                input_scaled = st.session_state['scaler'].transform(input_final)

                prediction = st.session_state['final_model'].predict(input_scaled)
                if st.session_state['problem_type'] == "Classification":
                    pred_label = st.session_state['le_target'].inverse_transform(prediction.astype(int))[0]
                    st.success(f"Predicted Performance: **{pred_label}**")
                else:
                    st.success(f"Predicted Value: **{prediction[0]:.2f}**")

        else:
            st.info("Train the model first to enable live prediction.")

# ---------- FOOTER ----------
st.sidebar.markdown("---")
st.sidebar.caption("• Student Habits Performance App")