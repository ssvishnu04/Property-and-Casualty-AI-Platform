import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from api_client import (
    check_api_health,
    explain_claim,
    predict_claim_risk,
)
from dashboard_utils import (
    format_currency,
    format_percent,
    load_champion_metadata,
    load_feature_drift_report,
    load_model_comparison,
    load_monitoring_summary,
    load_portfolio_summary,
    load_prediction_drift_report,
    load_prediction_logs,
    load_ragas_results,
    load_reinsurance_exposure,
    load_scored_claims,
)


st.set_page_config(
    page_title="Property & Casualty Domain Specialty AI Operations Console",
    page_icon="",
    layout="wide",
)


LOB_OPTIONS = [
    "Commercial Property",
    "General Liability",
    "Cyber",
    "Marine & Energy",
    "Professional Liability",
    "Workers Compensation",
    "Property Catastrophe Reinsurance",
    "Casualty Treaty Reinsurance",
]

LOSS_TYPE_OPTIONS = [
    "Hail",
    "Wind",
    "Fire",
    "Water Damage",
    "Flood",
    "Wildfire",
    "Hurricane",
    "Bodily Injury",
    "Ransomware",
    "Data Breach",
    "Large Liability Loss",
]

STATE_OPTIONS = [
    "TX", "FL", "CA", "NY", "IL", "GA", "LA", "NJ", "VA", "OH",
    "PA", "MI", "NC", "AZ", "WA", "MA", "IN", "CO", "TN", "MO",
]

TREATY_TYPE_OPTIONS = ["None", "Quota Share", "Excess of Loss", "Catastrophe XOL"]


def render_header():
    st.title("Property & Casualty Domain Specialty AI Operations Console")
    st.caption(
        "Production-style P&C and reinsurance AI platform: ML scoring, GenAI explanations, "
        "model monitoring, RAG evaluation, and audit visibility."
    )

    if check_api_health():
        st.success("FastAPI inference service is online")
    else:
        st.error(
            "FastAPI service is offline. Start it with: "
            "`uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`"
        )


def get_scenario_defaults(scenario: str) -> dict:
    scenarios = {
        "Reinsurance Triggered Claim": {
            "claim_id": "CLM-TEST-REINS-001",
            "policy_id": "POL-TEST-REINS-001",
            "treaty_id": "TRT-TEST-REINS-001",
            "line_of_business": "Commercial Property",
            "loss_type": "Hail",
            "state": "TX",
            "treaty_type": "Excess of Loss",
            "claim_amount": 425000.0,
            "reserve_amount": 500000.0,
            "prior_claim_count": 1,
            "litigation_flag": 0,
            "cat_exposure": 1,
            "suspicious_flag": 0,
            "retention": 250000.0,
            "treaty_limit": 2000000.0,
            "text_fraud_signal": 0,
            "text_litigation_signal": 0,
            "text_cat_signal": 1,
            "text_injury_signal": 0,
            "source_document_count": 2,
            "loss_year": 2026,
            "loss_month": 4,
            "loss_quarter": 2,
        },
        "Standard Low-Risk Claim": {
            "claim_id": "CLM-TEST-LOW-001",
            "policy_id": "POL-TEST-LOW-001",
            "treaty_id": "TRT-TEST-LOW-001",
            "line_of_business": "Commercial Property",
            "loss_type": "Water Damage",
            "state": "TX",
            "treaty_type": "None",
            "claim_amount": 15000.0,
            "reserve_amount": 18000.0,
            "prior_claim_count": 0,
            "litigation_flag": 0,
            "cat_exposure": 0,
            "suspicious_flag": 0,
            "retention": 999999999.0,
            "treaty_limit": 0.0,
            "text_fraud_signal": 0,
            "text_litigation_signal": 0,
            "text_cat_signal": 0,
            "text_injury_signal": 0,
            "source_document_count": 0,
            "loss_year": 2026,
            "loss_month": 2,
            "loss_quarter": 1,
        },
        "Fraud / SIU Review Claim": {
            "claim_id": "CLM-TEST-FRAUD-001",
            "policy_id": "POL-TEST-FRAUD-001",
            "treaty_id": "TRT-TEST-FRAUD-001",
            "line_of_business": "General Liability",
            "loss_type": "Bodily Injury",
            "state": "NY",
            "treaty_type": "Quota Share",
            "claim_amount": 120000.0,
            "reserve_amount": 150000.0,
            "prior_claim_count": 5,
            "litigation_flag": 1,
            "cat_exposure": 0,
            "suspicious_flag": 1,
            "retention": 250000.0,
            "treaty_limit": 2000000.0,
            "text_fraud_signal": 1,
            "text_litigation_signal": 1,
            "text_cat_signal": 0,
            "text_injury_signal": 1,
            "source_document_count": 3,
            "loss_year": 2026,
            "loss_month": 6,
            "loss_quarter": 2,
        },
        "Catastrophe Large Loss Claim": {
            "claim_id": "CLM-TEST-CAT-001",
            "policy_id": "POL-TEST-CAT-001",
            "treaty_id": "TRT-TEST-CAT-001",
            "line_of_business": "Property Catastrophe Reinsurance",
            "loss_type": "Hurricane",
            "state": "FL",
            "treaty_type": "Catastrophe XOL",
            "claim_amount": 1250000.0,
            "reserve_amount": 1400000.0,
            "prior_claim_count": 2,
            "litigation_flag": 0,
            "cat_exposure": 1,
            "suspicious_flag": 0,
            "retention": 500000.0,
            "treaty_limit": 5000000.0,
            "text_fraud_signal": 0,
            "text_litigation_signal": 0,
            "text_cat_signal": 1,
            "text_injury_signal": 0,
            "source_document_count": 4,
            "loss_year": 2026,
            "loss_month": 9,
            "loss_quarter": 3,
        },
    }

    if scenario == "Custom":
        return scenarios["Reinsurance Triggered Claim"]

    return scenarios.get(scenario, scenarios["Reinsurance Triggered Claim"])


def yes_no_select(label: str, default_value: int, help_text: str | None = None) -> int:
    return st.selectbox(
        label,
        [0, 1],
        index=int(default_value),
        format_func=lambda x: "Yes" if x == 1 else "No",
        help=help_text,
    )


def select_with_default(label: str, options: list[str], default_value: str, help_text: str | None = None):
    if default_value not in options:
        default_value = options[0]

    return st.selectbox(
        label,
        options,
        index=options.index(default_value),
        help=help_text,
    )


def build_claim_payload_from_sidebar() -> dict:
    st.sidebar.header("Real-Time Claim Setup")

    scenario = st.sidebar.selectbox(
        "Test Scenario",
        [
            "Reinsurance Triggered Claim",
            "Standard Low-Risk Claim",
            "Fraud / SIU Review Claim",
            "Catastrophe Large Loss Claim",
            "Custom",
        ],
        help="Choose a preset claim scenario, then adjust fields in the main form if needed.",
    )

    defaults = get_scenario_defaults(scenario)

    st.sidebar.caption(
        "Preset scenarios help users test common production cases without manually entering every field."
    )

    st.markdown("### Claim Input")

    with st.expander("1. Claim Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            claim_id = st.text_input(
                "Claim ID",
                defaults["claim_id"],
                help="Unique claim identifier from the claims system.",
            )

        with col2:
            policy_id = st.text_input(
                "Policy ID",
                defaults["policy_id"],
                help="Policy linked to the claim.",
            )

        with col3:
            treaty_id = st.text_input(
                "Treaty ID",
                defaults["treaty_id"],
                help="Reinsurance treaty linked to the policy or claim.",
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            line_of_business = select_with_default(
                "Line of Business",
                LOB_OPTIONS,
                defaults["line_of_business"],
                "Business segment such as Commercial Property, Cyber, or General Liability.",
            )

        with col5:
            loss_type = select_with_default(
                "Loss Type",
                LOSS_TYPE_OPTIONS,
                defaults["loss_type"],
                "Primary type of loss reported on the claim.",
            )

        with col6:
            state = select_with_default(
                "Loss State",
                STATE_OPTIONS,
                defaults["state"],
                "State where the loss occurred.",
            )

    with st.expander("2. Financial & Reinsurance Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            treaty_type = select_with_default(
                "Treaty Type",
                TREATY_TYPE_OPTIONS,
                defaults["treaty_type"],
                "Type of reinsurance arrangement. None means no reinsurance applies.",
            )

        with col2:
            claim_amount = st.number_input(
                "Claim Amount",
                min_value=0.0,
                value=float(defaults["claim_amount"]),
                step=10000.0,
                help="Reported claim amount or estimated loss amount.",
            )

        with col3:
            reserve_amount = st.number_input(
                "Current Reserve Amount",
                min_value=0.0,
                value=float(defaults["reserve_amount"]),
                step=10000.0,
                help="Current reserve set by claims team before ML recommendation.",
            )

        col4, col5 = st.columns(2)

        with col4:
            retention = st.number_input(
                "Reinsurance Retention",
                min_value=0.0,
                value=float(defaults["retention"]),
                step=50000.0,
                help="Amount the insurer retains before reinsurance pays.",
            )

        with col5:
            treaty_limit = st.number_input(
                "Treaty Limit",
                min_value=0.0,
                value=float(defaults["treaty_limit"]),
                step=100000.0,
                help="Maximum amount the reinsurer will pay under the treaty.",
            )

    with st.expander("3. Risk Indicators", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            prior_claim_count = st.number_input(
                "Prior Claim Count",
                min_value=0,
                value=int(defaults["prior_claim_count"]),
                step=1,
                help="Number of prior claims associated with the insured or policy.",
            )

        with col2:
            litigation_flag = yes_no_select(
                "Attorney / Litigation Involved?",
                defaults["litigation_flag"],
                "Yes when attorney involvement, litigation, or legal escalation exists.",
            )

        with col3:
            cat_exposure = yes_no_select(
                "Catastrophe Exposure?",
                defaults["cat_exposure"],
                "Yes when claim is related to hail, hurricane, flood, wildfire, windstorm, or other CAT event.",
            )

        with col4:
            suspicious_flag = yes_no_select(
                "Suspicious Indicators?",
                defaults["suspicious_flag"],
                "Yes when the claim has suspicious reporting, inconsistencies, or SIU indicators.",
            )

    with st.expander("4. Document / Text Signals", expanded=False):
        st.caption(
            "These simulate signals extracted from adjuster notes, emails, phone summaries, or ACORD documents."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            text_fraud_signal = yes_no_select(
                "Fraud Signal from Notes?",
                defaults["text_fraud_signal"],
                "Detected from notes/emails using keywords such as suspicious, inconsistent, late reporting, SIU, or fraud.",
            )

        with col2:
            text_litigation_signal = yes_no_select(
                "Attorney/Litigation Signal from Notes?",
                defaults["text_litigation_signal"],
                "Detected from notes/emails mentioning attorney, demand letter, lawsuit, legal review, or coverage dispute.",
            )

        with col3:
            text_cat_signal = yes_no_select(
                "CAT Signal from Notes?",
                defaults["text_cat_signal"],
                "Detected from notes/emails mentioning storm, hail, hurricane, wildfire, flood, or catastrophe.",
            )

        col4, col5 = st.columns(2)

        with col4:
            text_injury_signal = yes_no_select(
                "Injury Signal from Notes?",
                defaults["text_injury_signal"],
                "Detected from notes/emails mentioning injury, medical treatment, hospital, or bodily injury.",
            )

        with col5:
            source_document_count = st.number_input(
                "Related Source Documents",
                min_value=0,
                value=int(defaults["source_document_count"]),
                step=1,
                help="Number of notes, emails, XML notices, or supporting documents linked to the claim.",
            )

    with st.expander("5. Loss Timing", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            loss_year = st.number_input(
                "Loss Year",
                min_value=2000,
                max_value=2100,
                value=int(defaults["loss_year"]),
                step=1,
                help="Year the loss occurred.",
            )

        with col2:
            loss_month = st.slider(
                "Loss Month",
                1,
                12,
                int(defaults["loss_month"]),
                help="Month the loss occurred. Used for seasonality and drift analysis.",
            )

        with col3:
            loss_quarter = st.slider(
                "Loss Quarter",
                1,
                4,
                int(defaults["loss_quarter"]),
                help="Quarter the loss occurred. Used for seasonal grouping.",
            )

    return {
        "claim_id": claim_id,
        "policy_id": policy_id,
        "treaty_id": treaty_id,
        "line_of_business": line_of_business,
        "loss_type": loss_type,
        "state": state,
        "treaty_type": treaty_type,
        "claim_amount": claim_amount,
        "reserve_amount": reserve_amount,
        "prior_claim_count": prior_claim_count,
        "litigation_flag": litigation_flag,
        "cat_exposure": cat_exposure,
        "suspicious_flag": suspicious_flag,
        "retention": retention,
        "treaty_limit": treaty_limit,
        "text_fraud_signal": text_fraud_signal,
        "text_litigation_signal": text_litigation_signal,
        "text_cat_signal": text_cat_signal,
        "text_injury_signal": text_injury_signal,
        "source_document_count": source_document_count,
        "loss_year": loss_year,
        "loss_month": loss_month,
        "loss_quarter": loss_quarter,
    }


def clean_source_name(file_path: str) -> str:
    file_path = str(file_path).replace("\\", "/")
    return file_path.split("/")[-1].replace("_", " ").replace(".txt", "").title()


def render_operator_sources(sources: list[dict]):
    if not sources:
        st.write("No RAG sources retrieved.")
        return

    clean_rows = []

    for source in sources:
        file_path = source.get("file_path", "")
        clean_rows.append(
            {
                "Source Document": clean_source_name(file_path),
                "Source Path": file_path,
            }
        )

    st.table(pd.DataFrame(clean_rows))


def render_real_time_scoring_tab():
    st.subheader("Real-Time Claim Scoring")

    claim_payload = build_claim_payload_from_sidebar()

    question = st.text_area(
        "GenAI Question",
        value="Is this claim high risk, and what should the adjuster review?",
        height=90,
        help="Ask a business question about claim risk, reinsurance, fraud indicators, reserve, or adjuster next steps.",
    )

    run_prediction = st.button("Run ML + GenAI Analysis", type="primary")

    if not run_prediction:
        with st.expander("Submitted Claim Payload Preview", expanded=False):
            st.json(claim_payload)
        return

    try:
        prediction = predict_claim_risk(claim_payload)

        genai_payload = {
            "claim": claim_payload,
            "prediction": prediction,
            "question": question,
        }

        genai_response = explain_claim(genai_payload)

        st.markdown("### ML Prediction Summary")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        metric_col1.metric("Severity", prediction.get("severity_prediction"))
        metric_col2.metric(
            "Fraud Probability",
            format_percent(prediction.get("fraud_risk_probability")),
        )
        metric_col3.metric(
            "Recommended Reserve",
            format_currency(prediction.get("recommended_reserve")),
        )
        metric_col4.metric("Triage Priority", prediction.get("triage_priority"))

        st.markdown("### Reinsurance Impact")
        re_col1, re_col2, re_col3 = st.columns(3)

        re_col1.metric(
            "Retention Breach",
            "Yes" if prediction.get("retention_breach_flag") == 1 else "No",
        )
        re_col2.metric(
            "Ceded Loss",
            format_currency(prediction.get("calculated_ceded_loss")),
        )
        re_col3.metric(
            "Recovery Ratio",
            format_percent(prediction.get("reinsurance_recovery_ratio")),
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Business Risk Explanations")
            for item in prediction.get("risk_explanations", []):
                st.warning(item)

        with col2:
            st.markdown("### GenAI Explanation")
            st.info(genai_response.get("explanation"))

        st.markdown("### Retrieved Knowledge Sources")
        render_operator_sources(genai_response.get("retrieved_sources", []))

        with st.expander("Submitted Claim Payload", expanded=False):
            st.json(claim_payload)

    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


def render_portfolio_dashboard_tab():
    st.subheader("Portfolio Risk Dashboard")

    df = load_portfolio_summary()

    if df.empty:
        st.warning("Portfolio summary not found. Run Step 5 Gold feature engineering.")
        return

    total_claims = int(df["claim_count"].sum()) if "claim_count" in df.columns else 0
    total_claim_amount = df["total_claim_amount"].sum() if "total_claim_amount" in df.columns else 0
    total_high_priority = (
        df["high_priority_claim_count"].sum()
        if "high_priority_claim_count" in df.columns
        else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Claims", f"{total_claims:,}")
    col2.metric("Total Claim Amount", format_currency(total_claim_amount))
    col3.metric("High Priority Claims", f"{int(total_high_priority):,}")

    st.markdown("#### Portfolio Summary")
    st.table(df.head(50))

    if "line_of_business" in df.columns and "total_claim_amount" in df.columns:
        fig = px.bar(
            df,
            x="line_of_business",
            y="total_claim_amount",
            title="Total Claim Amount by Line of Business",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "line_of_business" in df.columns and "high_priority_rate" in df.columns:
        fig = px.bar(
            df,
            x="line_of_business",
            y="high_priority_rate",
            title="High Priority Rate by Line of Business",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This dashboard updates when the Gold feature engineering pipeline refreshes "
    )


def render_reinsurance_tab():
    st.subheader("Reinsurance Exposure Dashboard")

    df = load_reinsurance_exposure()

    if df.empty:
        st.warning("Reinsurance exposure file not found. Run Step 5 Gold feature engineering.")
        return

    total_ceded = df["total_ceded_loss"].sum() if "total_ceded_loss" in df.columns else 0
    total_claim_amount = df["total_claim_amount"].sum() if "total_claim_amount" in df.columns else 0
    breach_count = (
        df["retention_breach_count"].sum()
        if "retention_breach_count" in df.columns
        else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Ceded Loss", format_currency(total_ceded))
    col2.metric("Total Claim Amount", format_currency(total_claim_amount))
    col3.metric("Retention Breaches", f"{int(breach_count):,}")

    st.markdown("#### Reinsurance Exposure")
    st.table(df.head(50))

    if "line_of_business" in df.columns and "total_ceded_loss" in df.columns:
        fig = px.bar(
            df,
            x="line_of_business",
            y="total_ceded_loss",
            color="treaty_type" if "treaty_type" in df.columns else None,
            title="Total Ceded Loss by Line of Business and Treaty Type",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_prediction_audit_tab():
    st.subheader("Prediction Audit Log")

    df = load_prediction_logs()

    if df.empty:
        st.warning("No prediction logs found yet. Run a prediction first.")
        return

    df = df.sort_values("prediction_timestamp_utc", ascending=False)

    st.markdown("#### Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        triage_options = ["All"] + sorted(df["triage_priority"].dropna().unique().tolist())
        triage_filter = st.selectbox("Triage Priority", triage_options)

    with col2:
        max_rows = st.slider("Rows to display", 10, 500, 50)

    with col3:
        high_priority_only = st.checkbox("High priority only", value=False)

    filtered_df = df.copy()

    if triage_filter != "All":
        filtered_df = filtered_df[filtered_df["triage_priority"] == triage_filter]

    if high_priority_only and "high_priority_claim_flag" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["high_priority_claim_flag"] == 1]

    filtered_df = filtered_df.head(max_rows)

    st.table(filtered_df)

    if "triage_priority" in filtered_df.columns:
        fig = px.histogram(
            filtered_df,
            x="triage_priority",
            title="Prediction Count by Triage Priority",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "fraud_risk_probability" in filtered_df.columns:
        fig = px.histogram(
            filtered_df,
            x="fraud_risk_probability",
            nbins=20,
            title="Fraud Probability Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Prediction logs are append-only locally. In production, this would be partitioned "
        "by date in a Delta table or log analytics store with retention policies."
    )


def render_model_monitoring_tab():
    st.subheader("Model Monitoring")

    metadata = load_champion_metadata()

    if metadata:
        st.markdown("#### Champion Model Metadata")
        st.json(metadata)
    else:
        st.warning("Champion metadata not found.")

    comparison_df = load_model_comparison()

    if comparison_df.empty:
        st.warning("Model comparison summary not found.")
        return

    st.markdown("#### Model Comparison Results")
    st.table(comparison_df.head(100))

    metric_options = [
        col
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "mae", "rmse", "r2"]
        if col in comparison_df.columns
    ]

    if metric_options:
        selected_metric = st.selectbox("Select metric to compare", metric_options)

        fig = px.bar(
            comparison_df,
            x="model",
            y=selected_metric,
            color="target",
            title=f"Model Comparison by {selected_metric}",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_rag_evaluation_tab():
    st.subheader("RAG Evaluation Dashboard")

    df = load_ragas_results()

    if df.empty:
        st.warning("RAGAS results not found. Run Step 9 RAGAS evaluation.")
        return

    metric_cols = [
        col
        for col in [
            "faithfulness",
            "response_relevancy",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        if col in df.columns
    ]

    if metric_cols:
        avg_scores = df[metric_cols].mean(numeric_only=True).reset_index()
        avg_scores.columns = ["metric", "average_score"]

        st.markdown("#### Average RAGAS Scores")
        st.table(avg_scores)

        fig = px.bar(
            avg_scores,
            x="metric",
            y="average_score",
            title="Average RAGAS Metrics",
            range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed RAGAS Evaluation Results")
    st.table(df.head(100))

    st.caption(
        "RAGAS results are refreshed when the evaluation job runs, not on every prediction."
    )


def render_monitoring_drift_tab():
    st.subheader("Monitoring & Drift Dashboard")

    summary = load_monitoring_summary()
    feature_drift_df = load_feature_drift_report()
    prediction_drift_df = load_prediction_drift_report()

    if not summary:
        st.warning(
            "Monitoring summary not found"
        )
        return

    status = summary.get("status", "unknown")

    if status == "healthy":
        st.success("Monitoring status: Healthy")
    elif status == "review_required":
        st.error("Monitoring status: Review Required")
    else:
        st.warning(f"Monitoring status: {status}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Baseline Records", f"{summary.get('baseline_record_count', 0):,}")
    col2.metric("Current Prediction Records", f"{summary.get('current_prediction_record_count', 0):,}")
    col3.metric("Feature Drift Count", summary.get("feature_drift_count", 0))
    col4.metric("Prediction Drift Count", summary.get("prediction_drift_count", 0))

    st.caption(f"Last monitoring run: {summary.get('monitoring_timestamp_utc', 'N/A')}")

    st.markdown("#### Feature Drift Report")

    if feature_drift_df.empty:
        st.info("No feature drift report found.")
    else:
        st.table(feature_drift_df.head(100))

        if "drift_detected" in feature_drift_df.columns:
            drift_counts = (
                feature_drift_df["drift_detected"]
                .astype(str)
                .value_counts()
                .reset_index()
            )
            drift_counts.columns = ["drift_detected", "count"]

            fig = px.bar(
                drift_counts,
                x="drift_detected",
                y="count",
                title="Feature Drift Detected Count",
            )
            st.plotly_chart(fig, use_container_width=True)

        if "mean_change_pct" in feature_drift_df.columns:
            numeric_drift = feature_drift_df.dropna(subset=["mean_change_pct"])

            if not numeric_drift.empty:
                fig = px.bar(
                    numeric_drift,
                    x="column",
                    y="mean_change_pct",
                    title="Numeric Feature Mean Change %",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Prediction Drift Report")

    if prediction_drift_df.empty:
        st.info("No prediction drift report found.")
    else:
        st.table(prediction_drift_df.head(100))

        if "drift_detected" in prediction_drift_df.columns:
            pred_drift_counts = (
                prediction_drift_df["drift_detected"]
                .astype(str)
                .value_counts()
                .reset_index()
            )
            pred_drift_counts.columns = ["drift_detected", "count"]

            fig = px.bar(
                pred_drift_counts,
                x="drift_detected",
                y="count",
                title="Prediction Drift Detected Count",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This dashboard compares baseline Gold features and batch scored outputs "
        "against recent real-time prediction logs. In production, this would usually "
        "run on a schedule and feed alerting."
    )


def render_batch_scoring_tab():
    st.subheader("Batch Scored Claims Dashboard")

    df = load_scored_claims()

    if df.empty:
        st.warning(
            "Scored claims not found."
        )
        return

    total_scored = len(df)
    urgent_count = int((df["triage_priority"] == "Urgent").sum()) if "triage_priority" in df.columns else 0
    avg_fraud = df["fraud_risk_probability"].mean() if "fraud_risk_probability" in df.columns else 0
    total_reserve = df["recommended_reserve"].sum() if "recommended_reserve" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Scored Claims", f"{total_scored:,}")
    col2.metric("Urgent Claims", f"{urgent_count:,}")
    col3.metric("Avg Fraud Probability", format_percent(avg_fraud))
    col4.metric("Total Recommended Reserve", format_currency(total_reserve))

    st.markdown("#### Scored Claims Sample")
    display_cols = [
        col for col in [
            "claim_id",
            "line_of_business",
            "loss_type",
            "state",
            "claim_amount",
            "severity_prediction",
            "fraud_risk_probability",
            "recommended_reserve",
            "triage_priority",
            "batch_scoring_timestamp_utc",
        ]
        if col in df.columns
    ]

    st.table(df[display_cols].head(50))

    if "triage_priority" in df.columns:
        fig = px.histogram(
            df,
            x="triage_priority",
            title="Batch Scored Claims by Triage Priority",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "severity_prediction" in df.columns:
        fig = px.histogram(
            df,
            x="severity_prediction",
            title="Batch Scored Claims by Severity",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "fraud_risk_probability" in df.columns:
        fig = px.histogram(
            df,
            x="fraud_risk_probability",
            nbins=20,
            title="Fraud Probability Distribution - Batch Scoring",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "recommended_reserve" in df.columns and "line_of_business" in df.columns:
        reserve_by_lob = (
            df.groupby("line_of_business", as_index=False)["recommended_reserve"]
            .sum()
            .sort_values("recommended_reserve", ascending=False)
        )

        fig = px.bar(
            reserve_by_lob,
            x="line_of_business",
            y="recommended_reserve",
            title="Recommended Reserve by Line of Business",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top High-Risk Claims")

    high_risk_df = df.copy()

    if "triage_priority" in high_risk_df.columns:
        high_risk_df = high_risk_df[high_risk_df["triage_priority"] == "Urgent"]

    if "fraud_risk_probability" in high_risk_df.columns:
        high_risk_df = high_risk_df.sort_values(
            "fraud_risk_probability",
            ascending=False,
        )

    st.table(high_risk_df[display_cols].head(25))

    st.caption(
        "This tab is based on batch scoring output."
    )


def main():
    render_header()

    tabs = st.tabs(
        [
            "Real-Time Claim Scoring",
            "Batch Scored Claims",
            "Portfolio Risk",
            "Reinsurance Exposure",
            "Prediction Audit",
            "Model Monitoring",
            "RAG Evaluation",
            "Monitoring & Drift",
        ]
    )

    with tabs[0]:
        render_real_time_scoring_tab()

    with tabs[1]:
        render_batch_scoring_tab()

    with tabs[2]:
        render_portfolio_dashboard_tab()

    with tabs[3]:
        render_reinsurance_tab()

    with tabs[4]:
        render_prediction_audit_tab()

    with tabs[5]:
        render_model_monitoring_tab()

    with tabs[6]:
        render_rag_evaluation_tab()

    with tabs[7]:
        render_monitoring_drift_tab()


if __name__ == "__main__":
    main()