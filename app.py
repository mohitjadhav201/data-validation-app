import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Valuation Data Validator")
st.title(" Data Validation")

# Step 1: Select Benefit Type
benefit_type = st.radio("Choose a type to validate:", ("Gratuity", "Leave"), index=0)

# Initialize session state keys
if 'gratuity_inputs_saved' not in st.session_state:
    st.session_state.gratuity_inputs_saved = False

if benefit_type == "Gratuity":
    st.markdown("### 🧾 Gratuity Validation Setup")

    if not st.session_state.gratuity_inputs_saved:
        with st.form("gratuity_form"):
            company_name = st.text_input("Name of the Company")
            prepared_by = st.text_input("Prepared By")
            # Date inputs
            current_year = st.date_input("Current Date of Valuation")
            previous_valuation_date = st.date_input("Previous Date of Valuation")
            # Format dates as dd-mm-yyyy
            vesting_period = st.number_input("Vesting Period (in years)", min_value=0, max_value=10, value=5)
            cap_on_benefit = st.number_input("Cap on Benefit (₹)", min_value=0.0, value=2000000.0, step=50000.0)
            min_expected_salary_increase = st.number_input("Minimum Expected Increase in Salary (%)", min_value=0.0, max_value=0.0, value=0.0)
            max_expected_salary_increase = st.number_input("Maximum Expected Increase in Salary (%)", min_value=0.0, max_value=100.0, value=10.0)
            submitted = st.form_submit_button("Continue to Validation")

        if submitted:
            st.session_state.gratuity_inputs_saved = True
            st.session_state.gratuity_inputs = {
                'company_name': company_name,
                'prepared_by': prepared_by,
                'current_year': current_year,
                'previous_valuation_date': previous_valuation_date,
                'vesting_period': vesting_period,
                'cap_on_benefit': cap_on_benefit,
                'min_expected_salary_increase': min_expected_salary_increase,
                'max_expected_salary_increase': max_expected_salary_increase
            }

    else:
        inputs = st.session_state.gratuity_inputs
        st.success("✔ Inputs received successfully.")
        st.write("#### Summary of Inputs:")
        st.markdown(f"- **Company Name**: {inputs['company_name']}")
        st.markdown(f"- **Prepared By**: {inputs['prepared_by']}")
        st.markdown(f"- **Current Valuation Year**: {inputs['current_year'].strftime('%d-%m-%Y')}")
        st.markdown(f"- **Previous Valuation Date**: {inputs['previous_valuation_date'].strftime('%d-%m-%Y')}")
        st.markdown(f"- **Vesting Period**: {inputs['vesting_period']} years")
        st.markdown(f"- **Cap on Benefit**: ₹{int(inputs['cap_on_benefit']):,}")
        st.markdown(f"- **Max Expected Salary Increase**: {inputs['max_expected_salary_increase']}%")
        
        st.markdown("### 📂 Upload Excel Files")
        col1, col2 = st.columns(2)
        current_file = col1.file_uploader("Upload Current Data", type=["xlsx"], key="current")
        previous_file = col2.file_uploader("Upload Previous Data", type=["xlsx"], key="previous")
        if current_file and previous_file and st.button("▶ Run Validation"):
            st.markdown("### ✅ Running Current Data Validation...")
            try:
                df_current = pd.read_excel(current_file)
                df_previous = pd.read_excel(previous_file)
            except Exception as e:
                st.error(f"Error reading the files: {e}")
            else:
                df = df_current.copy()

                # Normalize columns
                df.columns = df.columns.str.strip()
                df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

                # Confirm structure
                expected_cols = [
                    'SR. NO.', 'EMPLOYEE CODE', 'D.O.B.', 'D.O.J.',
                    'Salary (Act)', 'Salary (Scheme)', 'Retirement Age'
                ]
                if not all(col in df.columns for col in expected_cols):
                    st.error("❌ Uploaded file does not have all required columns.")
                    st.write("Required columns:", expected_cols)
                    st.write("Uploaded file columns:", df.columns.tolist())
                else:
                    # Format valuation date
                    valuation_date_dt = pd.to_datetime(inputs['current_year'])
                    df['KAP Comments'] = ""

                    # 1. Missing values
                    for col in expected_cols:
                        missing_mask = df[col].isnull()
                        df.loc[missing_mask, 'KAP Comments'] += f"Missing value in {col}; "

                    # 2. Duplicate EMPLOYEE CODE (excluding blanks)
                    valid_codes = df['EMPLOYEE CODE'].notnull()
                    dupes = df[valid_codes].duplicated(subset='EMPLOYEE CODE', keep=False)
                    df.loc[valid_codes[valid_codes].index[dupes], 'KAP Comments'] += "Duplicate EMPLOYEE CODE; "


                    # 3. Salary validation
                    for col in ['Salary (Act)', 'Salary (Scheme)']:
                        low_salary = df[col] < 1000
                        df.loc[low_salary, 'KAP Comments'] += f"{col} less than 1000; "

                    # 4. D.O.B. > D.O.J.
                    df['D.O.B.'] = pd.to_datetime(df['D.O.B.'], errors='coerce')
                    df['D.O.J.'] = pd.to_datetime(df['D.O.J.'], errors='coerce')
                    dob_after_doj = df['D.O.B.'] > df['D.O.J.']
                    df.loc[dob_after_doj, 'KAP Comments'] += "D.O.B. is after D.O.J.; "

                    # 5. Age at valuation
                    df['Age at Valuation'] = ((valuation_date_dt - df['D.O.B.']).dt.days // 365)
                    age_exceed = df['Age at Valuation'] > df['Retirement Age']
                    df.loc[age_exceed, 'KAP Comments'] += "Actual Age exceeds Retirement Age; "

                    # 6. Filter error rows
                    error_rows_df = df[df['KAP Comments'] != ""]

                    if error_rows_df.empty:
                        st.success("✅ No validation errors found in the current file.")
                    else:
                        st.error("⚠ Validation errors found:")
                        st.dataframe(error_rows_df[expected_cols + ['KAP Comments']], use_container_width=True)

                        # Download error report
                        st.download_button("📥 Download Error Report (CSV)",
                                        data=error_rows_df.to_csv(index=False),
                                        file_name="current_validation_errors.csv",
                                        mime="text/csv")
                                # ---------------------- COMPARISON ANALYSIS ----------------------
                        # ========== COMPARISON ANALYSIS ==========
            st.markdown("## 🔍 Comparison Analysis")

            # Format dynamic date suffixes
            current_date_suffix = inputs['current_year'].strftime('%d-%m-%Y')
            previous_date_suffix = inputs['previous_valuation_date'].strftime('%d-%m-%Y')
            safe_current_suffix = current_date_suffix.replace('-', '_')
            safe_previous_suffix = previous_date_suffix.replace('-', '_')

            # 1. Count Summary
            st.write(f"**Current Data Count:** {len(df_current)}")
            st.write(f"**Previous Data Count:** {len(df_previous)}")

            # Standardize column names
            df_current.columns = df_current.columns.str.strip()
            df_previous.columns = df_previous.columns.str.strip()

            # Merge dataframes on EMPLOYEE CODE
            merged_df = pd.merge(
                df_current,
                df_previous,
                on='EMPLOYEE CODE',
                how='outer',
                suffixes=(f'_{safe_current_suffix}', f'_{safe_previous_suffix}'),
                indicator=True
            )

            # 2. Unexplained Employees
            # 2. Unexplained Employees (based on D.O.J. < Previous Valuation Date)
            unexplained_mask = (merged_df['_merge'] == 'left_only') & (
                pd.to_datetime(merged_df[f'D.O.J._{safe_current_suffix}'], errors='coerce') < pd.to_datetime(inputs['previous_valuation_date'])
            )

            current_cols = [col for col in merged_df.columns if col.endswith(f'_{safe_current_suffix}') or col == 'EMPLOYEE CODE']
            unexplained_df = merged_df[unexplained_mask][current_cols].copy()
            unexplained_df.columns = [col.replace(f'_{safe_current_suffix}', '') for col in unexplained_df.columns]
            unexplained_df['KAP Comments'] = "Unexplained Employee – Not present in previous year"

            if not unexplained_df.empty:
                st.subheader("🧩 Unexplained Employees")
                st.dataframe(unexplained_df, use_container_width=True)
                st.download_button("📥 Download Unexplained Employees", unexplained_df.to_csv(index=False), "unexplained_employees.csv", "text/csv")
                
            # 3. D.O.B. Mismatches
            dob_col_current = f"D.O.B._{safe_current_suffix}"
            dob_col_previous = f"D.O.B._{safe_previous_suffix}"

            dob_mismatch = merged_df[
                (merged_df['_merge'] == 'both') &
                (pd.to_datetime(merged_df[dob_col_current], errors='coerce') != pd.to_datetime(merged_df[dob_col_previous], errors='coerce'))
            ].copy()

            if not dob_mismatch.empty:
                dob_mismatch[f'D.O.B. {current_date_suffix}'] = pd.to_datetime(dob_mismatch[dob_col_current], errors='coerce').dt.strftime('%d-%m-%Y')
                dob_mismatch[f'D.O.B. {previous_date_suffix}'] = pd.to_datetime(dob_mismatch[dob_col_previous], errors='coerce').dt.strftime('%d-%m-%Y')
                dob_mismatch['Difference in Days'] = (
                    pd.to_datetime(dob_mismatch[dob_col_current], errors='coerce') -
                    pd.to_datetime(dob_mismatch[dob_col_previous], errors='coerce')
                ).dt.days

                st.subheader("📆 D.O.B. Mismatches")
                st.dataframe(
                    dob_mismatch[['EMPLOYEE CODE', f"EMPLOYEE'S NAME_{safe_current_suffix}", f'D.O.B. {current_date_suffix}', f'D.O.B. {previous_date_suffix}', 'Difference in Days']],
                    use_container_width=True
                )

            # 4. D.O.J. Mismatches
            doj_col_current = f"D.O.J._{safe_current_suffix}"
            doj_col_previous = f"D.O.J._{safe_previous_suffix}"

            doj_mismatch = merged_df[
                (merged_df['_merge'] == 'both') &
                (pd.to_datetime(merged_df[doj_col_current], errors='coerce') != pd.to_datetime(merged_df[doj_col_previous], errors='coerce'))
            ].copy()

            if not doj_mismatch.empty:
                doj_mismatch[f'D.O.J. {current_date_suffix}'] = pd.to_datetime(doj_mismatch[doj_col_current], errors='coerce').dt.strftime('%d-%m-%Y')
                doj_mismatch[f'D.O.J. {previous_date_suffix}'] = pd.to_datetime(doj_mismatch[doj_col_previous], errors='coerce').dt.strftime('%d-%m-%Y')
                doj_mismatch['Difference in Days'] = (
                    pd.to_datetime(doj_mismatch[doj_col_current], errors='coerce') -
                    pd.to_datetime(doj_mismatch[doj_col_previous], errors='coerce')
                ).dt.days

                st.subheader("🛬 D.O.J. Mismatches")
                st.dataframe(
                    doj_mismatch[['EMPLOYEE CODE', f"EMPLOYEE'S NAME_{safe_current_suffix}", f'D.O.J. {current_date_suffix}', f'D.O.J. {previous_date_suffix}', 'Difference in Days']],
                    use_container_width=True
                )

            # 5. Salary Inconsistencies
            salary_col_current = f"Salary (Act)_{safe_current_suffix}"
            salary_col_previous = f"Salary (Act)_{safe_previous_suffix}"

            salary_mismatch = merged_df[
                (merged_df['_merge'] == 'both') &
                (merged_df[salary_col_current].notnull()) &
                (merged_df[salary_col_previous].notnull())
            ].copy()

            # 🛠 Convert salary columns to numeric
            salary_mismatch[salary_col_current] = pd.to_numeric(salary_mismatch[salary_col_current], errors='coerce')
            salary_mismatch[salary_col_previous] = pd.to_numeric(salary_mismatch[salary_col_previous], errors='coerce')

            # 💰 Calculate percentage increase
            salary_mismatch['% Increase'] = (
                (salary_mismatch[salary_col_current] - salary_mismatch[salary_col_previous]) /
                salary_mismatch[salary_col_previous]
            ) * 100

            max_inc = inputs['max_expected_salary_increase']
            min_inc = -max_inc

            salary_flagged = salary_mismatch[
                (salary_mismatch['% Increase'] > max_inc) |
                (salary_mismatch['% Increase'] < min_inc)
            ]

            if not salary_flagged.empty:
                st.subheader("💰 Salary Inconsistencies (Beyond Thresholds)")
                st.dataframe(
                    salary_flagged[['EMPLOYEE CODE', f"EMPLOYEE'S NAME_{safe_current_suffix}", salary_col_current, salary_col_previous, '% Increase']],
                    use_container_width=True
                )
            st.markdown("### 📄 Current Employee Data")
            st.dataframe(df_current, use_container_width=True)
            # ========== EMPLOYEE STATUS & BENEFIT ANALYSIS SECTION ==========
            st.markdown("## 📊 Employee Status & Benefit Analysis")

            try:
                previous_df = df_previous.copy()
                previous_df.columns = previous_df.columns.str.strip()

                # Prepare data
                analysis_df = previous_df.copy()
                analysis_df['D.O.B.'] = pd.to_datetime(analysis_df['D.O.B.'], errors='coerce')
                analysis_df['D.O.J.'] = pd.to_datetime(analysis_df['D.O.J.'], errors='coerce')
                analysis_df['Salary (Act)'] = pd.to_numeric(analysis_df['Salary (Act)'], errors='coerce')

                # Merge current salary to calculate increase
                analysis_df = pd.merge(
                    analysis_df,
                    df_current[['EMPLOYEE CODE', 'Salary (Act)']],
                    on='EMPLOYEE CODE',
                    how='left',
                    suffixes=('', '_Current')
                )
                analysis_df['Increase in Salary %'] = ((analysis_df['Salary (Act)_Current'] - analysis_df['Salary (Act)']) / analysis_df['Salary (Act)']) * 100
                analysis_df.drop(columns='Salary (Act)_Current', inplace=True)

                # Separation Status as Label
                current_emp_codes = df_current['EMPLOYEE CODE'].unique()
                analysis_df['Employment Status'] = np.where(
                    analysis_df['EMPLOYEE CODE'].isin(current_emp_codes), 'Present', 'Separated'
                )

                # Calculate exact age and past service
                prev_val_date = pd.to_datetime(inputs['previous_valuation_date'])
                analysis_df['Exact Age Opening'] = ((prev_val_date - analysis_df['D.O.B.']).dt.days / 365).round(2)
                analysis_df['Exact Past Service Opening'] = ((prev_val_date - analysis_df['D.O.J.']).dt.days / 365)

                # Round past service for payment calculation
                analysis_df['Rounded Past Service'] = analysis_df['Exact Past Service Opening'].round(0)

                # Cap on Benefit
                cap = inputs['cap_on_benefit']
                vesting = inputs['vesting_period']

                # Payment on Separation (capped if separated)
                analysis_df['Payment on Separation'] = np.where(
                    analysis_df['Employment Status'] == 'Separated',
                    np.minimum((15 / 26) * analysis_df['Rounded Past Service'] * analysis_df['Salary (Act)'], cap),
                    0
                )

                # Vested Payment on Separation
                analysis_df['Vested Payment on Separation'] = np.where(
                    (analysis_df['Employment Status'] == 'Separated') & (analysis_df['Exact Past Service Opening'] >= vesting),
                    np.minimum((15 / 26) * analysis_df['Rounded Past Service'] * analysis_df['Salary (Act)'], cap),
                    0
                )

                # Display columns
                display_cols = ['EMPLOYEE CODE', "EMPLOYEE'S NAME", 'Salary (Act)', 'Increase in Salary %',
                                'Employment Status', 'Exact Age Opening', 'Exact Past Service Opening',
                                'Payment on Separation', 'Vested Payment on Separation']
                
                st.dataframe(analysis_df[display_cols], use_container_width=True)

                # Download Button
                st.download_button(
                    "📥 Download Analysis Report",
                    analysis_df[display_cols].to_csv(index=False),
                    file_name="employee_status_analysis.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"🚨 Error during analysis: {e}")
            # Filter for Vested Separated Employees with non-zero payment
            vested_df = analysis_df[
                (analysis_df['Employment Status'] == 'Separated') &
                (analysis_df['Exact Past Service Opening'] >= vesting) &
                (analysis_df['Vested Payment on Separation'] > 0)
            ]

            st.markdown("### ✅ Vested Separated Employees with Payments")
            st.dataframe(vested_df[display_cols], use_container_width=True)

            # Summary Table
            total_payment = analysis_df.loc[analysis_df['Employment Status'] == 'Separated', 'Payment on Separation'].sum()
            total_vested = vested_df['Vested Payment on Separation'].sum()

            summary_data = pd.DataFrame({
                "Metric": ["Total Payment (All Separated)", "Total Vested Payment", "Non-Vested"],
                "Amount": [round(total_payment, 2), round(total_vested, 2), round(total_payment - total_vested, 2)]
            })

            st.markdown("### 📊 Summary of Payments")
            st.dataframe(summary_data, use_container_width=True)

            # Download Vested Payment Data (optional)
            st.download_button(
                "📥 Download Vested Employees Report",
                vested_df[display_cols].to_csv(index=False),
                file_name="vested_employees_payment.csv",
                mime="text/csv"
            )
elif benefit_type == "Leave":
    st.warning("🛠 Leave Validation UI will be added in the next phase.")
