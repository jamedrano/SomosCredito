import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OrdinalEncoder
from scipy.stats import chi2_contingency, f_oneway

# --- Core Risk Score Calculation Logic ---
def calculate_risk_score_df(df_input, grace_period_days, weights):
    if df_input.empty: return None
    df = df_input.copy()
    required_base_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion', 'credito', 'reglaCobranza', 'cuotaEsperada', 'totalTrans', 'saldoCapitalActual', 'totalDesembolso', 'cobranzaTrans', 'categoriaProductoCrediticio']
    required_cols = required_base_cols 
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: st.error(f"Sheet 'HistoricoPagoCuotas' missing columns: {', '.join(missing_cols)}"); return None
    date_cols = ['fechaDesembolso', 'fechaEsperadaPago', 'fechaPagoRecibido', 'fechaRegistro', 'fechaTRansaccion']
    for col in date_cols:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
    df['credito'] = df['credito'].astype(str)
    grace_period = pd.Timedelta(days=grace_period_days)
    mask_valid_dates_late_payment = df['fechaPagoRecibido'].notna() & df['fechaEsperadaPago'].notna()
    df['late_payment'] = 0
    df.loc[mask_valid_dates_late_payment, 'late_payment'] = (df.loc[mask_valid_dates_late_payment, 'fechaPagoRecibido'] > (df.loc[mask_valid_dates_late_payment, 'fechaEsperadaPago'] + grace_period)).astype(int)
    late_payment_counts = df.groupby('credito')['late_payment'].sum()
    total_payments_due_count = df.groupby('credito')['fechaEsperadaPago'].count()
    late_payment_ratio = (late_payment_counts / total_payments_due_count.replace(0, np.nan)).fillna(0)
    df['totalTrans'] = pd.to_numeric(df['totalTrans'], errors='coerce').fillna(0) 
    total_payment_made_monetary = df.groupby('credito')['totalTrans'].sum() 
    df['cuotaEsperada'] = pd.to_numeric(df['cuotaEsperada'], errors='coerce').fillna(0) 
    total_payment_expected_monetary = df.groupby('credito')['cuotaEsperada'].sum()
    payment_coverage_ratio = (total_payment_made_monetary / total_payment_expected_monetary.replace(0, np.nan)).fillna(1).replace([np.inf, -np.inf], 1) 
    df['saldoCapitalActual'] = pd.to_numeric(df['saldoCapitalActual'], errors='coerce')
    df['totalDesembolso'] = pd.to_numeric(df['totalDesembolso'], errors='coerce')
    last_saldo = df.groupby('credito')['saldoCapitalActual'].last()
    first_desembolso = df.groupby('credito')['totalDesembolso'].first()
    outstanding_balance_ratio = (last_saldo / first_desembolso.replace(0, np.nan)).fillna(0)
    df['cobranzaTrans'] = pd.to_numeric(df['cobranzaTrans'], errors='coerce').fillna(0)
    df['collection_activity'] = (df['cobranzaTrans'] > 0).astype(int)
    collection_activity_count = df.groupby('credito')['collection_activity'].sum()
    creditos_unicos = pd.DataFrame(df['credito'].unique(), columns=['credito'])
    creditos_unicos = creditos_unicos.merge(late_payment_ratio.to_frame(name='late_payment_ratio'), left_on='credito', right_index=True, how='left').fillna({'late_payment_ratio': 0})
    creditos_unicos = creditos_unicos.merge(payment_coverage_ratio.to_frame(name='payment_coverage_ratio'), left_on='credito', right_index=True, how='left').fillna({'payment_coverage_ratio': 1})
    creditos_unicos = creditos_unicos.merge(outstanding_balance_ratio.to_frame(name='outstanding_balance_ratio'), left_on='credito', right_index=True, how='left').fillna({'outstanding_balance_ratio': 0})
    creditos_unicos = creditos_unicos.merge(collection_activity_count.to_frame(name='collection_activity_count'), left_on='credito', right_index=True, how='left').fillna({'collection_activity_count': 0})
    df_for_return = creditos_unicos.copy()
    component_cols_to_scale = ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
    for col_name in component_cols_to_scale:
        min_val, max_val = creditos_unicos[col_name].min(), creditos_unicos[col_name].max()
        scaled_col_name = f'{col_name}_scaled'
        if max_val == min_val: df_for_return[scaled_col_name] = 0.0
        else: df_for_return[scaled_col_name] = (creditos_unicos[col_name] - min_val) / (max_val - min_val)
        df_for_return[scaled_col_name] = df_for_return[scaled_col_name].fillna(0)
    df_for_return['risk_score'] = (weights['late_payment_ratio'] * df_for_return['late_payment_ratio_scaled'] + weights['payment_coverage_ratio'] * (1 - df_for_return['payment_coverage_ratio_scaled']) + weights['outstanding_balance_ratio'] * df_for_return['outstanding_balance_ratio_scaled'] + weights['collection_activity_count'] * df_for_return['collection_activity_count_scaled'])
    cols_to_return = ['credito', 'risk_score', 'late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
    return df_for_return[cols_to_return]

# --- Utility Function for Outlier Detection ---
def get_outliers_iqr(df, column_name):
    if df is None or df.empty or column_name not in df.columns or df[column_name].isnull().all(): return pd.DataFrame(), pd.DataFrame(), np.nan, np.nan
    Q1, Q3 = df[column_name].quantile(0.25), df[column_name].quantile(0.75); IQR = Q3 - Q1
    if IQR == 0: lower_bound, upper_bound = Q1, Q3
    else: lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[df[column_name] < lower_bound], df[df[column_name] > upper_bound], lower_bound, upper_bound

# --- Helper function for Cramer's V ---
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n; r, k = confusion_matrix.shape
    if r == 1 or k == 1 or n == 0: return 0
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)); rcorr = r - ((r-1)**2)/(n-1); kcorr = k - ((k-1)**2)/(n-1)
    if rcorr < 1 or kcorr < 1 : return 0 
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# --- Data Preparation for PRE-LOAN FEATURE INSIGHTS TAB (Tab 4) ---
@st.cache_data
def prepare_preloan_insights_data(risk_df_with_components, listado_df, id_col_listado, target_col_name_in_risk_scores, selected_features_from_listado, bin_target_flag, num_bins_for_target=3):
    if risk_df_with_components is None or risk_df_with_components.empty or listado_df is None or listado_df.empty:
        return None, None, None, None, None, "Risk scores or customer data not available." # Added one None for original_dtypes
    if id_col_listado not in listado_df.columns:
        return None, None, None, None, None, f"ID column '{id_col_listado}' not found in customer data."
    if target_col_name_in_risk_scores not in risk_df_with_components.columns:
        return None, None, None, None, None, f"Target column '{target_col_name_in_risk_scores}' not found in risk scores."

    listado_copy = listado_df.copy()
    risk_scores_copy = risk_df_with_components.copy()
    listado_copy[id_col_listado] = listado_copy[id_col_listado].astype(str)
    risk_scores_copy['credito'] = risk_scores_copy['credito'].astype(str)
    
    merged_df = pd.merge(
        listado_copy, # Take all columns from listado for now
        risk_scores_copy[['credito', target_col_name_in_risk_scores]], # Only target and ID from risk scores
        left_on=id_col_listado,
        right_on='credito',
        how='inner'
    )
    if merged_df.empty:
        return None, None, None, None, None, "No matching records found after merging."

    # Filter for selected features from listado ONLY
    features_to_analyze = [f for f in selected_features_from_listado if f in merged_df.columns and f not in [target_col_name_in_risk_scores, id_col_listado, 'credito']]
    if not features_to_analyze:
        return None, None, None, None, None, "No valid features selected/found from ListadoCreditos for analysis."

    # Create a DataFrame with only the features to analyze and the target
    analysis_subset_df = merged_df[features_to_analyze + [target_col_name_in_risk_scores]].copy()
    
    # Prepare y (target series)
    y_series = analysis_subset_df[target_col_name_in_risk_scores].copy()
    final_target_name = target_col_name_in_risk_scores # Default target name
    if bin_target_flag:
        if y_series.nunique() <= 1:
            return None, None, None, None, None, f"Target '{target_col_name_in_risk_scores}' has <=1 unique value, cannot bin."
        try:
            discretizer = KBinsDiscretizer(n_bins=num_bins_for_target, encode='ordinal', strategy='quantile', subsample=None)
            binned_target_values = discretizer.fit_transform(y_series.to_frame())
            y_series = pd.Series(binned_target_values.ravel().astype(int).astype(str), index=y_series.index) # Make it categorical string
            final_target_name = target_col_name_in_risk_scores + '_binned'
            y_series.name = final_target_name # Update series name
        except ValueError as e:
            return None, None, None, None, None, f"Error binning target '{target_col_name_in_risk_scores}': {e}."
            
    # Prepare X (features DataFrame) and store original dtypes
    x_features_df = pd.DataFrame(index=analysis_subset_df.index)
    original_dtypes = {}
    for feature in features_to_analyze:
        original_dtypes[feature] = analysis_subset_df[feature].dtype # Get dtype from the subset
        if pd.api.types.is_numeric_dtype(analysis_subset_df[feature]):
            x_features_df[feature] = analysis_subset_df[feature].fillna(analysis_subset_df[feature].median())
        elif pd.api.types.is_object_dtype(analysis_subset_df[feature]) or pd.api.types.is_categorical_dtype(analysis_subset_df[feature]):
            mode_val = analysis_subset_df[feature].mode()
            impute_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            x_features_df[feature] = analysis_subset_df[feature].fillna(impute_val)
        else:
            x_features_df[feature] = analysis_subset_df[feature].astype(str).fillna("Unknown")
            
    return x_features_df, y_series, features_to_analyze, original_dtypes, final_target_name, None


# --- Data Preparation for FEATURE MI RANKER TAB (Tab 6) ---
@st.cache_data
def prepare_mi_data(risk_scores_data, listado_data, id_col_listado, target_col_name_in_risk_scores, selected_features_from_listado, bin_target_flag, num_bins_for_target=3):
    if risk_scores_data is None or risk_scores_data.empty or listado_data is None or listado_data.empty:
        return None, None, None, None, "Risk scores or customer data not available for MI calculation."
    if id_col_listado not in listado_data.columns:
        return None, None, None, None, f"ID column '{id_col_listado}' not found in customer data."
    if target_col_name_in_risk_scores not in risk_scores_data.columns:
        return None, None, None, None, f"Target column '{target_col_name_in_risk_scores}' not found in risk scores."

    listado_copy = listado_data.copy(); risk_scores_copy = risk_scores_data.copy()
    listado_copy[id_col_listado] = listado_copy[id_col_listado].astype(str)
    risk_scores_copy['credito'] = risk_scores_copy['credito'].astype(str)
    
    merged_df_for_mi = pd.merge(listado_copy, risk_scores_copy[['credito', target_col_name_in_risk_scores]], left_on=id_col_listado, right_on='credito', how='inner')
    if merged_df_for_mi.empty: return None, None, None, None, "No matching records found after merging."

    features_to_process = [f for f in selected_features_from_listado if f in merged_df_for_mi.columns]
    if not features_to_process: return None, None, None, None, "No valid features selected/found from ListadoCreditos for MI."

    X_mi = pd.DataFrame(index=merged_df_for_mi.index)
    discrete_mask = [] 
    processed_feature_names_ordered = [] # Keep track of order and actual processed names

    for feature in features_to_process:
        col_data = merged_df_for_mi[feature].copy()
        if pd.api.types.is_numeric_dtype(col_data):
            X_mi[feature] = col_data.fillna(col_data.median())
            discrete_mask.append(col_data.nunique(dropna=False) < 20) # Consider NaNs as a category for nunique check if not dropping
            processed_feature_names_ordered.append(feature)
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            mode_val = col_data.mode(); impute_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            col_data_filled = col_data.fillna(impute_val).astype(str)
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) # Or LabelEncoder if 1D
            X_mi[feature] = encoder.fit_transform(col_data_filled.to_frame())
            discrete_mask.append(True)
            processed_feature_names_ordered.append(feature)
        else: # Skip other types for MI or handle explicitly if needed
            st.warning(f"Skipping feature '{feature}' for MI due to unhandled data type: {col_data.dtype}")
            continue 
    
    if X_mi.empty or not processed_feature_names_ordered : return None, None, None, None, "No features were processed for MI."

    y_mi_target_name = target_col_name_in_risk_scores
    y_mi = merged_df_for_mi[target_col_name_in_risk_scores].copy()
    if bin_target_flag:
        if y_mi.nunique() <= 1: return None, None, None, None, f"Target '{y_mi_target_name}' has <=1 unique value, cannot bin."
        try:
            discretizer = KBinsDiscretizer(n_bins=num_bins_for_target, encode='ordinal', strategy='quantile', subsample=None)
            y_mi_binned = discretizer.fit_transform(y_mi.to_frame())
            y_mi = pd.Series(y_mi_binned.ravel().astype(int), index=y_mi.index)
        except ValueError as e: return None, None, None, None, f"Error binning target '{y_mi_target_name}' for MI: {e}."
    else: y_mi = pd.to_numeric(y_mi, errors='coerce').fillna(y_mi.median())
            
    return X_mi[processed_feature_names_ordered], y_mi, processed_feature_names_ordered, discrete_mask, None


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🏍️ Motorcycle Loan Risk & Data Insights App")
risk_score_component_names = ['late_payment_ratio', 'payment_coverage_ratio', 'outstanding_balance_ratio', 'collection_activity_count']
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
default_grace_period = 5
grace_period_input = st.sidebar.number_input("Grace Period (days)", min_value=0, max_value=30, value=default_grace_period, step=1)
st.sidebar.subheader("Indicator Weights")
w_late_payment = st.sidebar.slider("Late Payment Ratio Weight", 0.0, 1.0, 0.40, 0.01)
w_payment_coverage = st.sidebar.slider("Payment Coverage Ratio Weight", 0.0, 1.0, 0.40, 0.01)
w_outstanding_balance = st.sidebar.slider("Outstanding Balance Ratio Weight", 0.0, 1.0, 0.20, 0.01)
w_collection_activity = st.sidebar.slider("Collection Activity Count Weight", 0.0, 1.0, 0.00, 0.01)
user_weights = {'late_payment_ratio': w_late_payment, 'payment_coverage_ratio': w_payment_coverage, 'outstanding_balance_ratio': w_outstanding_balance, 'collection_activity_count': w_collection_activity}
risk_scores_df, listado_creditos_df = None, None
processed_data_info, historico_pago_cuotas_loaded, listado_creditos_loaded = "", False, False
numero_credito_col_name = "numeroCredito"

if uploaded_file:
    st.sidebar.info(f"Processing: {uploaded_file.name}")
    try:
        hpc_df = pd.read_excel(uploaded_file, sheet_name="HistoricoPagoCuotas")
        processed_data_info += "✅ 'HistoricoPagoCuotas' loaded.\n"; historico_pago_cuotas_loaded = True
        if 'categoriaProductoCrediticio' in hpc_df.columns:
            moto_df = hpc_df[hpc_df["categoriaProductoCrediticio"] == "MOTOS"].copy()
            if not moto_df.empty:
                processed_data_info += f"Found {len(moto_df)} 'MOTOS' records.\n";
                with st.spinner("Calculating risk scores & components..."): risk_scores_df = calculate_risk_score_df(moto_df, grace_period_input, user_weights)
                if risk_scores_df is not None and not risk_scores_df.empty: processed_data_info += f"✅ Risk scores & components for {len(risk_scores_df)} credits.\n"
                else: processed_data_info += "⚠️ Risk score calculation issues.\n"
            else: processed_data_info += "⚠️ No 'MOTOS' data.\n"
        else: processed_data_info += "❌ 'categoriaProductoCrediticio' missing.\n"; historico_pago_cuotas_loaded = False
    except Exception as e: processed_data_info += f"❌ Error 'HistoricoPagoCuotas': {e}\n"; historico_pago_cuotas_loaded = False
    try:
        lc_df_temp = pd.read_excel(uploaded_file, sheet_name="ListadoCreditos")
        processed_data_info += f"✅ 'ListadoCreditos' loaded ({lc_df_temp.shape[0]}r, {lc_df_temp.shape[1]}c).\n"; listado_creditos_loaded = True
        if numero_credito_col_name in lc_df_temp.columns:
            lc_df_temp[numero_credito_col_name] = lc_df_temp[numero_credito_col_name].astype(str)
            listado_creditos_df = lc_df_temp; processed_data_info += f"'{numero_credito_col_name}' cast.\n"
        else: processed_data_info += f"⚠️ '{numero_credito_col_name}' missing.\n"; listado_creditos_df = lc_df_temp
    except Exception as e: processed_data_info += f"❌ Error 'ListadoCreditos': {e}\n"; listado_creditos_loaded = False
    st.sidebar.text_area("File Processing Log", processed_data_info, height=200)
else: st.info("☝️ Upload an Excel file to begin.")

tab_titles = ["📊 Risk Scores", "📈 Risk EDA", "🕵️ Outlier Analysis", "📋 Customer Data Quality", "🔍 Pre-Loan Insights", "📊 Segment Performance", "ℹ️ Feature MI Ranker"]
tabs = st.tabs(tab_titles)

with tabs[0]: # Risk Scores
    st.header(tab_titles[0])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Calculated Risk Scores & Components (per Credit)")
        display_cols_scores = ['credito', 'risk_score'] + risk_score_component_names
        style_format_dict_tab0 = {"risk_score": "{:.4f}", "late_payment_ratio": "{:.4f}", "payment_coverage_ratio": "{:.4f}", "outstanding_balance_ratio": "{:.4f}", "collection_activity_count": "{:.0f}"}
        st.dataframe(risk_scores_df[display_cols_scores].style.format(style_format_dict_tab0), height=500, use_container_width=True)
        output_tab0 = io.BytesIO()
        with pd.ExcelWriter(output_tab0, engine='xlsxwriter') as writer_tab0: risk_scores_df.to_excel(writer_tab0, index=False, sheet_name='RiskScoresAndComponents')
        excel_data_tab0 = output_tab0.getvalue()
        if excel_data_tab0: st.download_button(label="📥 Download Scores & Components", data=excel_data_tab0, file_name=f"risk_scores_components_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.warning("Could not generate Excel file for download.")
    elif uploaded_file and historico_pago_cuotas_loaded: st.warning("Risk scores not calculated. Check log.")
    elif uploaded_file and not historico_pago_cuotas_loaded: st.error("'HistoricoPagoCuotas' failed to load.")
    else: st.write("Upload file for results.")

with tabs[1]: # Risk EDA
    st.header(tab_titles[1])
    if risk_scores_df is not None and not risk_scores_df.empty:
        st.subheader("Risk Score Distribution"); st.dataframe(risk_scores_df['risk_score'].describe().to_frame().T.style.format("{:.4f}"))
        col1, col2 = st.columns(2)
        with col1: fig, ax = plt.subplots(); sns.histplot(risk_scores_df['risk_score'], kde=True, ax=ax, bins=20); ax.set_title('Histogram'); st.pyplot(fig); plt.close(fig)
        with col2: fig, ax = plt.subplots(); sns.boxplot(y=risk_scores_df['risk_score'], ax=ax); ax.set_title('Boxplot'); st.pyplot(fig); plt.close(fig)
    elif uploaded_file: st.warning("Risk scores unavailable for EDA.")
    else: st.write("Upload file for EDA.")

with tabs[2]: # Outlier Analysis
    st.header(tab_titles[2])
    if risk_scores_df is not None and not risk_scores_df.empty:
        low_o, high_o, lb, ub = get_outliers_iqr(risk_scores_df, 'risk_score')
        st.subheader("Risk Score Outlier ID (IQR)");
        if not np.isnan(lb): st.write(f"Bounds: {lb:.4f} - {ub:.4f}")
        else: st.write("Cannot determine outlier bounds.")
        st.markdown("---"); st.subheader("High-Risk Outliers")
        if not high_o.empty: st.write(f"Found: {len(high_o)}"); st.dataframe(high_o['risk_score'].describe().to_frame().T.style.format("{:.4f}"));
        else: st.write("No high-risk outliers.")
        st.markdown("---"); st.subheader("Low-Risk Outliers")
        if not low_o.empty: st.write(f"Found: {len(low_o)}"); st.dataframe(low_o['risk_score'].describe().to_frame().T.style.format("{:.4f}"));
        else: st.write("No low-risk outliers.")
        st.markdown("---"); st.subheader("Download Outlier Details")
        if listado_creditos_loaded and listado_creditos_df is not None and numero_credito_col_name in listado_creditos_df.columns:
            if not high_o.empty or not low_o.empty:
                output_o = io.BytesIO()
                with pd.ExcelWriter(output_o, engine='xlsxwriter') as writer:
                    if not high_o.empty:
                        high_o_details = listado_creditos_df.merge(high_o, left_on=numero_credito_col_name, right_on='credito', how='inner')
                        if 'credito_y' in high_o_details.columns : high_o_details = high_o_details.drop(columns=['credito_y'])
                        if 'credito_x' in high_o_details.columns : high_o_details = high_o_details.rename(columns={'credito_x':'credito'})
                        high_o_details.to_excel(writer, sheet_name='High Risk', index=False)
                    if not low_o.empty:
                        low_o_details = listado_creditos_df.merge(low_o, left_on=numero_credito_col_name, right_on='credito', how='inner')
                        if 'credito_y' in low_o_details.columns : low_o_details = low_o_details.drop(columns=['credito_y'])
                        if 'credito_x' in low_o_details.columns : low_o_details = low_o_details.rename(columns={'credito_x':'credito'})
                        low_o_details.to_excel(writer, sheet_name='Low Risk', index=False)
                excel_data_outlier = output_o.getvalue()
                if excel_data_outlier: st.download_button(label="📥 Download Outlier Details", data=excel_data_outlier, file_name=f"outliers_detailed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else: st.warning("Could not generate outlier details Excel file.")
            else: st.info("No outliers to download.")
        elif uploaded_file: st.warning(f"'ListadoCreditos' or '{numero_credito_col_name}' issue. Check log.")
    elif uploaded_file: st.warning("Risk scores unavailable.")
    else: st.write("Upload file for analysis.")

with tabs[3]: # Customer Data Quality
    st.header(tab_titles[3] + " ('ListadoCreditos')")
    if not uploaded_file: st.write("Upload file for DQA.")
    elif not listado_creditos_loaded or listado_creditos_df is None: st.error("'ListadoCreditos' not loaded. Check log.")
    else:
        df_dqa = listado_creditos_df
        st.subheader("1. Overview"); st.write(f"Rows: {df_dqa.shape[0]}, Columns: {df_dqa.shape[1]}");
        with st.expander("Data Types"): st.dataframe(df_dqa.dtypes.reset_index().rename(columns={'index':'Col',0:'Type'}))
        st.subheader("2. Missing Values"); missing_sum = df_dqa.isnull().sum().reset_index(); missing_sum.columns=['Col','Missing']; missing_sum['%']=(missing_sum['Missing']/len(df_dqa))*100
        missing_sum = missing_sum[missing_sum['Missing']>0].sort_values(by='%',ascending=False)
        if not missing_sum.empty: st.dataframe(missing_sum.style.format({'%':"{:.2f}%"}));
        else: st.success("No missing values! 🎉")
        st.subheader("3. Duplicates"); st.write(f"Full Duplicates: {df_dqa.duplicated().sum()}")
        if numero_credito_col_name in df_dqa.columns: st.write(f"'{numero_credito_col_name}' Duplicates: {df_dqa.duplicated(subset=[numero_credito_col_name]).sum()}")
        st.subheader("4. Column Analysis")
        default_dqa_col = [df_dqa.columns[0]] if len(df_dqa.columns) > 0 else []
        cols_detail = st.multiselect("Select columns for detail:", options=df_dqa.columns.tolist(), default=default_dqa_col)
        for col in cols_detail:
            with st.expander(f"'{col}' (Type: {df_dqa[col].dtype})"):
                st.write(f"Unique: {df_dqa[col].nunique()}, Missing: {df_dqa[col].isnull().sum()} ({df_dqa[col].isnull().sum()/len(df_dqa)*100:.2f}%)")
                if pd.api.types.is_numeric_dtype(df_dqa[col]): st.dataframe(df_dqa[col].describe().to_frame().T)
                elif pd.api.types.is_object_dtype(df_dqa[col]): st.dataframe(df_dqa[col].value_counts().nlargest(10).reset_index())

with tabs[4]: # Pre-Loan Insights
    st.header(tab_titles[4])
    if not uploaded_file: st.write("Upload an Excel file and ensure 'ListadoCreditos' and risk scores are processed.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty: st.warning("Customer data ('ListadoCreditos') or Risk Scores are not available. Please check previous tabs/logs.")
    else:
        st.subheader("Configuration for Pre-Loan Feature Insights")
        target_options_prev_tab = ["Raw Risk Score (Continuous)", "Binned Risk Score (Categorical)"]
        chosen_target_type_prev_tab = st.selectbox("How to treat Risk Score for analysis (Pre-Loan Insights)?", target_options_prev_tab, index=0, key="fi_target_type")
        num_bins_fi_prev_tab = 3
        if "Binned" in chosen_target_type_prev_tab: num_bins_fi_prev_tab = st.slider("Number of bins for Risk Score (Pre-Loan Insights):", 2, 10, 3, 1, key="fi_bins")
        
        available_features_prev_tab = [col for col in listado_creditos_df.columns if col not in [numero_credito_col_name]]
        if not available_features_prev_tab: st.error("No features from 'ListadoCreditos' for Pre-Loan Insights.")
        else:
            default_fi_col_prev_tab = [available_features_prev_tab[0]] if len(available_features_prev_tab) > 0 else []
            selected_cols_fi_prev_tab = st.multiselect("Select features for Pre-Loan Insights:", options=available_features_prev_tab, default=default_fi_col_prev_tab, key="fi_cols")
            
            if st.button("🚀 Analyze Pre-Loan Insights", key="fi_analyze_button"):
                if not selected_cols_fi_prev_tab: st.warning("Please select at least one feature for Pre-Loan Insights.")
                else:
                    with st.spinner("Preparing data and performing pre-loan feature analysis..."):
                        # Using prepare_preloan_insights_data for this tab
                        prep_result_tab4 = prepare_preloan_insights_data(
                            risk_scores_df, listado_creditos_df, numero_credito_col_name, 
                            'risk_score', selected_cols_fi_prev_tab, 
                            "Binned" in chosen_target_type_prev_tab, num_bins_fi_prev_tab
                        )
                        
                        # Unpack results for Tab 4
                        features_for_analysis_df_tab4, target_series_tab4, actual_features_analyzed_tab4, \
                        original_feature_dtypes_tab4, final_target_name_tab4, error_message_tab4 = prep_result_tab4

                        if error_message_tab4: st.error(f"Data Prep Error (Pre-Loan Insights): {error_message_tab4}")
                        elif features_for_analysis_df_tab4 is None or target_series_tab4 is None: st.error("Failed to prepare data for Pre-Loan Insights.")
                        else:
                            st.success(f"Data prepared for Pre-Loan Insights. Analyzing {len(actual_features_analyzed_tab4)} features against '{final_target_name_tab4}'.")
                            
                            # A. Correlation Analysis (for Tab 4)
                            if "Raw" in chosen_target_type_prev_tab: # Check if target is continuous for correlation
                                st.markdown("---"); st.subheader("A. Correlation with Raw Risk Score")
                                numeric_features_to_correlate_tab4 = [f for f in actual_features_analyzed_tab4 if pd.api.types.is_numeric_dtype(original_feature_dtypes_tab4.get(f)) and f in selected_cols_fi_prev_tab]
                                if numeric_features_to_correlate_tab4:
                                    correlations_tab4 = {feat: features_for_analysis_df_tab4[feat].corr(target_series_tab4, method='pearson') if pd.api.types.is_numeric_dtype(target_series_tab4) else np.nan for feat in numeric_features_to_correlate_tab4}
                                    corr_df_tab4 = pd.DataFrame.from_dict(correlations_tab4, orient='index', columns=['Pearson Correlation']).dropna()
                                    if not corr_df_tab4.empty:
                                        corr_df_display_tab4 = corr_df_tab4.copy(); corr_df_display_tab4['Abs Correlation'] = corr_df_display_tab4['Pearson Correlation'].abs(); corr_df_display_tab4 = corr_df_display_tab4.sort_values(by='Abs Correlation', ascending=False)
                                        st.write("Correlation Table:"); st.dataframe(corr_df_display_tab4[['Pearson Correlation']].style.format("{:.3f}"))
                                        st.subheader("Scatter Plots")
                                        for feat_name in corr_df_display_tab4.index: 
                                            if feat_name in features_for_analysis_df_tab4.columns and pd.api.types.is_numeric_dtype(target_series_tab4):
                                                fig_corr, ax_corr = plt.subplots(); sns.regplot(x=features_for_analysis_df_tab4[feat_name], y=target_series_tab4, ax=ax_corr, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}); ax_corr.set_title(f"Scatter: {feat_name} vs. {final_target_name_tab4}"); st.pyplot(fig_corr); plt.close(fig_corr)
                                    else: st.info("No valid correlations.")
                                else: st.info("No numeric features selected for correlation.")
                            
                            # B. Group-wise Comparisons (for Tab 4)
                            st.markdown("---"); st.subheader("B. Group-wise Comparisons")
                            categorical_features_to_group_tab4 = [f for f in actual_features_analyzed_tab4 if (not pd.api.types.is_numeric_dtype(original_feature_dtypes_tab4.get(f)) or features_for_analysis_df_tab4[f].nunique() < 20) and f in selected_cols_fi_prev_tab]
                            if categorical_features_to_group_tab4:
                                results_groupwise_tab4 = []
                                for feat in categorical_features_to_group_tab4:
                                    if features_for_analysis_df_tab4[feat].nunique() < 2 or features_for_analysis_df_tab4[feat].nunique() > 50 : continue
                                    plot_target_groupwise_tab4 = pd.to_numeric(target_series_tab4, errors='coerce') if "Raw" in chosen_target_type_prev_tab else target_series_tab4
                                    try: order = (features_for_analysis_df_tab4.groupby(feat)[final_target_name_tab4].median().sort_values().index.astype(str) if "Raw" in chosen_target_type_prev_tab and pd.api.types.is_numeric_dtype(plot_target_groupwise_tab4) and features_for_analysis_df_tab4[feat].nunique() > 1 else sorted(features_for_analysis_df_tab4[feat].dropna().unique().astype(str)))
                                    except Exception: order = sorted(features_for_analysis_df_tab4[feat].dropna().unique().astype(str))
                                    if len(order) > 0 and not plot_target_groupwise_tab4.isnull().all():
                                        fig_box_fi, ax_box_fi = plt.subplots(); sns.boxplot(x=features_for_analysis_df_tab4[feat].astype(str), y=plot_target_groupwise_tab4, ax=ax_box_fi, order=order); ax_box_fi.set_title(f"{final_target_name_tab4} by {feat}"); ax_box_fi.tick_params(axis='x', rotation=45); plt.tight_layout(); st.pyplot(fig_box_fi); plt.close(fig_box_fi)
                                    if "Binned" in chosen_target_type_prev_tab:
                                        ct = pd.crosstab(features_for_analysis_df_tab4[feat], target_series_tab4);
                                        if ct.size > 0 and 0 not in ct.shape and ct.sum().sum() > 0:
                                            try: chi2, p, _, _ = chi2_contingency(ct); cv = cramers_v(ct.values); results_groupwise_tab4.append({'Feature': feat, 'Test': 'Chi2', 'Stat': chi2, 'p': p, "CramerV": cv})
                                            except ValueError: pass 
                                    else:
                                        groups = [target_series_tab4[features_for_analysis_df_tab4[feat] == cat] for cat in features_for_analysis_df_tab4[feat].unique() if target_series_tab4[features_for_analysis_df_tab4[feat] == cat].shape[0] > 1]
                                        if len(groups) > 1:
                                            try: f_stat, p_anova = f_oneway(*groups); results_groupwise_tab4.append({'Feature': feat, 'Test': 'ANOVA', 'Stat': f_stat, 'p': p_anova, "CramerV": np.nan})
                                            except ValueError: pass 
                                if results_groupwise_tab4: st.dataframe(pd.DataFrame(results_groupwise_tab4).style.format({'Stat': "{:.3f}", 'p': "{:.3g}", "CramerV": "{:.3f}"}))
                                else: st.info("No group tests for selected categories.")
                            else: st.info("No categorical features selected for group comparison.")

                            # C. Mutual Information (for Tab 4)
                            st.markdown("---"); st.subheader("C. Mutual Information")
                            mi_features_to_analyze_tab4 = [f for f in actual_features_analyzed_tab4 if f in selected_cols_fi_prev_tab]
                            if mi_features_to_analyze_tab4:
                                mi_features_df_tab4 = pd.DataFrame(index=features_for_analysis_df_tab4.index); 
                                label_encoders_tab4 = {} # For potential inverse transform if needed, not used here
                                discrete_mask_tab4 = []
                                for feat in mi_features_to_analyze_tab4: # Process only selected features
                                    if not pd.api.types.is_numeric_dtype(original_feature_dtypes_tab4.get(feat)): 
                                        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                                        mi_features_df_tab4[feat] = le.fit_transform(features_for_analysis_df_tab4[[feat]]) # Use original data from features_for_analysis_df_tab4
                                        discrete_mask_tab4.append(True)
                                    else: 
                                        mi_features_df_tab4[feat] = features_for_analysis_df_tab4[feat].fillna(features_for_analysis_df_tab4[feat].median())
                                        discrete_mask_tab4.append(mi_features_df_tab4[feat].nunique()<20)

                                if not mi_features_df_tab4.empty:
                                    mi_target_tab4 = target_series_tab4.astype(int) if "Binned" in chosen_target_type_prev_tab else pd.to_numeric(target_series_tab4, errors='coerce').fillna(target_series_tab4.median())
                                    mi_func_tab4 = mutual_info_classif if "Binned" in chosen_target_type_prev_tab else mutual_info_regression
                                    
                                    effective_discrete_mask_tab4 = discrete_mask_tab4 if len(discrete_mask_tab4) == mi_features_df_tab4.shape[1] else 'auto'

                                    mi_scores_tab4 = mi_func_tab4(mi_features_df_tab4, mi_target_tab4, discrete_features=effective_discrete_mask_tab4, random_state=42)
                                    mi_df_tab4 = pd.DataFrame({'Feature': mi_features_to_analyze_tab4, 'MI': mi_scores_tab4}).sort_values(by='MI', ascending=False)
                                    st.dataframe(mi_df_tab4.style.format({'MI': "{:.4f}"}))
                                    if not mi_df_tab4.empty: 
                                        fig_mi, ax_mi = plt.subplots(figsize=(10, max(5, len(mi_df_tab4)*0.3))); sns.barplot(x='MI',y='Feature',data=mi_df_tab4,ax=ax_mi); ax_mi.set_title("Mutual Information (Pre-Loan Insights)"); plt.tight_layout(); st.pyplot(fig_mi); plt.close(fig_mi)
                                else: st.info("No features for MI from selection.")
                            else: st.info("No features selected for MI.")


with tabs[5]: # Segment Performance Analyzer
    st.header(tab_titles[5]) 
    if not uploaded_file: st.write("Upload an Excel file and ensure 'ListadoCreditos' and risk scores are processed.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty: st.warning("Customer data ('ListadoCreditos') or Risk Scores (with components) are not available. Please check processing.")
    else:
        st.subheader("Define Customer Segment")
        temp_listado_df = listado_creditos_df.copy(); temp_risk_scores_df = risk_scores_df.copy()
        temp_listado_df[numero_credito_col_name] = temp_listado_df[numero_credito_col_name].astype(str)
        temp_risk_scores_df['credito'] = temp_risk_scores_df['credito'].astype(str)
        required_risk_cols_for_segment = ['credito', 'risk_score'] + risk_score_component_names
        if not all(col in temp_risk_scores_df.columns for col in required_risk_cols_for_segment): st.error(f"Risk score data missing required columns: {', '.join(required_risk_cols_for_segment)}")
        else:
            segment_data_full = pd.merge(temp_listado_df, temp_risk_scores_df[required_risk_cols_for_segment], left_on=numero_credito_col_name, right_on='credito', how='inner')
            if segment_data_full.empty: st.warning("No matching records found between customer data and risk scores for segmentation.")
            else:
                demographic_cols = [col for col in listado_creditos_df.columns if col not in [numero_credito_col_name] + required_risk_cols_for_segment]
                categorical_demographics = [col for col in demographic_cols if segment_data_full[col].dtype == 'object' or segment_data_full[col].nunique() < 20]
                if not categorical_demographics: st.info("No suitable categorical demographic variables found in 'ListadoCreditos'.")
                else:
                    selected_segment_vars = st.multiselect("Select demographic variables for segmentation:", options=categorical_demographics, default=categorical_demographics[0] if categorical_demographics else [])
                    filters = {}
                    for var in selected_segment_vars:
                        unique_levels = sorted(segment_data_full[var].dropna().unique().astype(str))
                        if unique_levels: 
                            default_level_selection = [unique_levels[0]] if unique_levels else [] 
                            filters[var] = st.multiselect(f"Select levels for '{var}':", options=unique_levels, default=default_level_selection)
                        else: st.caption(f"No selectable levels for '{var}'.")
                    
                    segmented_df = segment_data_full.copy()
                    if filters:
                        query_parts = []
                        for var, levels in filters.items():
                            if levels:
                                str_levels_for_query = []
                                for level_item in levels:
                                    escaped_level_item = str(level_item).replace("'", "\\'") 
                                    str_levels_for_query.append(f"'{escaped_level_item}'")
                                if str_levels_for_query:
                                    query_parts.append(f"`{var}` in ({', '.join(str_levels_for_query)})")
                        if query_parts:
                            try: segmented_df = segmented_df.query(" and ".join(query_parts))
                            except Exception as e: st.error(f"Error applying filters: {e}. Check column/level names for special chars."); segmented_df = pd.DataFrame()
                    
                    if not segmented_df.empty:
                        st.markdown("---"); st.subheader(f"Performance for Selected Segment ({len(segmented_df)} loans)")
                        avg_risk_score_segment = segmented_df['risk_score'].mean()
                        avg_components_segment = segmented_df[risk_score_component_names].mean().to_dict()
                        segment_summary_list = [{'Metric': 'Risk Score', 'Segment Average': avg_risk_score_segment}]
                        for comp_name in risk_score_component_names: segment_summary_list.append({'Metric': comp_name, 'Segment Average': avg_components_segment.get(comp_name, np.nan)})
                        segment_summary_df = pd.DataFrame(segment_summary_list)
                        def format_value_segment_tab(val, metric_name):
                            if pd.isna(val): return "N/A"
                            if metric_name == 'Risk Score': return f"{val:.4f}"
                            if 'ratio' in metric_name : return f"{val:.4f}" 
                            if 'count' in metric_name : return f"{val:.2f}"
                            return f"{val:.4f}" 
                        segment_summary_df['Segment Average Formatted'] = segment_summary_df.apply(lambda row: format_value_segment_tab(row['Segment Average'], row['Metric']), axis=1)
                        st.dataframe(segment_summary_df[['Metric', 'Segment Average Formatted']].set_index('Metric'))
                        with st.expander("Compare with Overall Portfolio Averages"):
                            avg_risk_score_overall = risk_scores_df['risk_score'].mean()
                            avg_components_overall = risk_scores_df[risk_score_component_names].mean().to_dict()
                            overall_summary_list = [{'Metric': 'Risk Score', 'Overall Average': avg_risk_score_overall}]
                            for comp_name in risk_score_component_names: overall_summary_list.append({'Metric': comp_name, 'Overall Average': avg_components_overall.get(comp_name, np.nan)})
                            overall_summary_df = pd.DataFrame(overall_summary_list)
                            comparison_df = pd.merge(segment_summary_df[['Metric','Segment Average']], overall_summary_df, on="Metric")
                            for col_to_format in ['Segment Average', 'Overall Average']: comparison_df[col_to_format + ' Formatted'] = comparison_df.apply(lambda row: format_value_segment_tab(row[col_to_format], row['Metric']), axis=1 )
                            st.dataframe(comparison_df[['Metric', 'Segment Average Formatted', 'Overall Average Formatted']].set_index('Metric'))
                    elif filters and not query_parts and any(filters.values()): st.info("Please select specific levels for the chosen demographic variables.")
                    elif filters and query_parts and segmented_df.empty : st.info("No customers found matching all selected criteria.")
                    else: st.info("Select demographic variables and their levels to analyze segment performance.")

with tabs[6]: # Feature MI Ranker
    st.header(tab_titles[6])
    if not uploaded_file: st.write("Upload an Excel file and ensure 'ListadoCreditos' and risk scores are processed.")
    elif listado_creditos_df is None or risk_scores_df is None or risk_scores_df.empty: st.warning("Customer data ('ListadoCreditos') or Risk Scores are not available.")
    else:
        st.subheader("Configuration for Mutual Information Ranking")
        mi_target_options = ["Raw Risk Score (Continuous)", "Binned Risk Score (Categorical)"]
        mi_chosen_target_type = st.selectbox("Treat Risk Score as:", mi_target_options, index=0, key="mi_ranker_target_type")
        mi_num_bins = 3
        if "Binned" in mi_chosen_target_type:
            mi_num_bins = st.slider("Number of bins for Risk Score (MI Ranker):", 2, 10, 3, 1, key="mi_ranker_bins")
        mi_available_features = [col for col in listado_creditos_df.columns if col not in [numero_credito_col_name]]
        if not mi_available_features: st.error("No features available from 'ListadoCreditos' for MI ranking.")
        else:
            default_mi_cols = mi_available_features[:min(5, len(mi_available_features))] 
            selected_cols_for_mi = st.multiselect("Select features from 'ListadoCreditos' for MI calculation:", options=mi_available_features, default=default_mi_cols, key="mi_ranker_features")
            if st.button("📈 Calculate Mutual Information", key="mi_ranker_button"):
                if not selected_cols_for_mi: st.warning("Please select at least one feature for MI calculation.")
                else:
                    with st.spinner("Preparing data and calculating Mutual Information..."):
                        X_mi_prepared, y_mi_prepared, processed_names_mi, discrete_feature_mask_mi, mi_error_message = prepare_mi_data(risk_scores_df, listado_creditos_df, numero_credito_col_name, 'risk_score', selected_cols_for_mi, "Binned" in mi_chosen_target_type, mi_num_bins)
                        if mi_error_message: st.error(f"MI Data Preparation Error: {mi_error_message}")
                        elif X_mi_prepared is None or y_mi_prepared is None or not processed_names_mi: st.error("Failed to prepare data for MI calculation. Check selected features or data integrity.")
                        else:
                            st.success(f"Data prepared. Calculating MI for {len(processed_names_mi)} features.")
                            mi_function_to_use = mutual_info_classif if "Binned" in mi_chosen_target_type else mutual_info_regression
                            effective_discrete_mask_mi = discrete_feature_mask_mi if len(discrete_feature_mask_mi) == X_mi_prepared.shape[1] else 'auto'
                            mi_scores_values = mi_function_to_use(X_mi_prepared, y_mi_prepared, discrete_features=effective_discrete_mask_mi, random_state=42)
                            mi_results_df = pd.DataFrame({'Feature': processed_names_mi, 'Mutual Information Score': mi_scores_values}).sort_values(by='Mutual Information Score', ascending=False)
                            st.subheader("Mutual Information Scores with Risk Score")
                            st.dataframe(mi_results_df.style.format({'Mutual Information Score': "{:.4f}"}))
                            if not mi_results_df.empty:
                                fig_mi_ranker, ax_mi_ranker = plt.subplots(figsize=(10, max(5, len(mi_results_df) * 0.3))); sns.barplot(x='Mutual Information Score', y='Feature', data=mi_results_df, ax=ax_mi_ranker, palette="viridis"); ax_mi_ranker.set_title("Feature Ranking by MI with Risk Score"); plt.tight_layout(); st.pyplot(fig_mi_ranker); plt.close(fig_mi_ranker)
st.markdown("---")
st.markdown("App developed by your Expert Data Scientist, Antonio Medrano, CepSA")
