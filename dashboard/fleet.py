
import streamlit as st
import pandas as pd
import textwrap

from database.db import get_all_checkins


def show_fleet_dashboard():

    # =========================================
    # LOAD DATABASE RECORDS
    # =========================================

    records = get_all_checkins()
    df = pd.DataFrame(records)

    expected_columns = [
        "timestamp",
        "driver_id",
        "bpm",
        "quality_score",
        "status",
        "reasons"
    ]

    if df.empty:
        df = pd.DataFrame(columns=expected_columns)

    for column in expected_columns:
        if column not in df.columns:
            df[column] = None

    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["quality_score"] = pd.to_numeric(
        df["quality_score"], errors="coerce"
    )

    df["status_normalized"] = (
        df["status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # =========================================
    # DASHBOARD HEADER DESIGN
    # =========================================

    st.html(textwrap.dedent("""
    <style>
    .fleet-header {
        background: linear-gradient(110deg, #172B42, #123C42);
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #28545A;
        margin-bottom: 20px;
    }

    .fleet-eyebrow {
        color: #55E6C1;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .fleet-title {
        font-size: 32px;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .fleet-subtitle {
        color: #C1D2DF;
        font-size: 14px;
    }
    </style>
    """))

    st.html("""
    <div class="fleet-header">
        <div class="fleet-eyebrow">FLEET INTELLIGENCE</div>
        <div class="fleet-title">Fleet Analytics Dashboard</div>
        <div class="fleet-subtitle">
            Monitor driver check-ins and measurement quality.
        </div>
    </div>
    """)

    # =========================================
    # REFRESH DASHBOARD
    # =========================================

    refresh_col1, refresh_col2 = st.columns([5, 1])

    with refresh_col1:
        st.caption(
            "Refresh to load the latest records saved in the database."
        )

    with refresh_col2:
        if st.button(
            "🔄 Refresh",
            use_container_width=True
        ):
            st.rerun()

    # =========================================
    # CALCULATE DASHBOARD METRICS
    # =========================================

    total = len(df)

    accepted = (
        df["status_normalized"] == "accepted"
    ).sum()

    rejected = (
        df["status_normalized"] == "rejected"
    ).sum()

    test_records = (
        df["status_normalized"] == "test"
    ).sum()

    completed = accepted + rejected

    acceptance_rate = (
        accepted / completed * 100
        if completed > 0 else 0
    )

    average_quality = df["quality_score"].mean()

    # =========================================
    # FLEET OVERVIEW
    # =========================================

    st.markdown("## 📊 Fleet Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Check-ins", total)

    with col2:
        st.metric("Accepted", accepted)

    with col3:
        st.metric("Rejected", rejected)

    with col4:
        st.metric(
            "Acceptance Rate",
            f"{acceptance_rate:.1f}%"
        )

    col5, col6 = st.columns(2)

    with col5:
        quality_display = (
            f"{average_quality:.1f}"
            if pd.notna(average_quality)
            else "—"
        )

        st.metric(
            "Average Quality Score",
            quality_display
        )

    with col6:
        st.metric("Test Records", test_records)

    st.caption(
        "Acceptance rate uses accepted and rejected measurements only. "
        "Demo records are excluded."
    )

    st.divider()

    # =========================================
    # SEARCH AND FILTER
    # =========================================

    st.markdown("## 🔎 Search and Filter Records")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        search_driver = st.text_input(
            "Search Driver ID",
            placeholder="Enter Driver ID..."
        )

    with filter_col2:
        selected_status = st.selectbox(
            "Filter by measurement status",
            ["All", "Accepted", "Rejected", "Test"]
        )

    filtered_df = df.copy()

    if search_driver.strip():
        filtered_df = filtered_df[
            filtered_df["driver_id"]
            .fillna("")
            .astype(str)
            .str.contains(
                search_driver.strip(),
                case=False,
                na=False
            )
        ]

    if selected_status != "All":
        filtered_df = filtered_df[
            filtered_df["status_normalized"]
            == selected_status.lower()
        ]

    st.caption(
        f"Showing {len(filtered_df)} of {total} check-in records."
    )

    # =========================================
    # DRIVER CHECK-IN RECORDS
    # =========================================

    st.markdown("## 👥 Driver Check-in Records")

    if filtered_df.empty:

        st.info(
            "No check-in records match your current search or filter."
        )

    else:

        display_df = filtered_df[expected_columns].copy()

        display_df = display_df.rename(columns={
            "timestamp": "Timestamp",
            "driver_id": "Driver ID",
            "bpm": "Heart Rate (BPM)",
            "quality_score": "Quality Score",
            "status": "Status",
            "reasons": "Remarks"
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = display_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Check-in Records",
            data=csv_data,
            file_name="pulsecam_fleet_records.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()

    # =========================================
    # HEART RATE ANALYTICS
    # =========================================

    st.markdown("## ❤️ Heart Rate Analytics")

    heart_rate_df = filtered_df.dropna(
        subset=["bpm"]
    ).copy()

    if not heart_rate_df.empty:

        heart_rate_summary = (
            heart_rate_df
            .groupby("driver_id", dropna=False)["bpm"]
            .mean()
            .reset_index()
        )

        heart_rate_summary["driver_id"] = (
            heart_rate_summary["driver_id"]
            .fillna("Unknown")
            .astype(str)
        )

        heart_rate_summary = heart_rate_summary.rename(
            columns={
                "driver_id": "Driver ID",
                "bpm": "Average BPM"
            }
        )

        st.bar_chart(
            heart_rate_summary.set_index("Driver ID"),
            y="Average BPM"
        )

    else:
        st.info(
            "Heart-rate analytics will appear when actual BPM "
            "measurements are saved."
        )

    st.divider()

    # =========================================
    # SIGNAL QUALITY ANALYTICS
    # =========================================

    st.markdown("## 📡 Signal Quality Analytics")

    quality_df = filtered_df.dropna(
        subset=["quality_score"]
    ).copy()

    if not quality_df.empty:

        quality_summary = (
            quality_df
            .groupby("driver_id", dropna=False)["quality_score"]
            .mean()
            .reset_index()
        )

        quality_summary["driver_id"] = (
            quality_summary["driver_id"]
            .fillna("Unknown")
            .astype(str)
        )

        quality_summary = quality_summary.rename(
            columns={
                "driver_id": "Driver ID",
                "quality_score": "Average Quality"
            }
        )

        st.bar_chart(
            quality_summary.set_index("Driver ID"),
            y="Average Quality"
        )

    else:
        st.info(
            "Signal quality analytics will appear when actual "
            "quality scores are saved."
        )

    st.divider()

    # =========================================
    # REJECTION DETAILS
    # =========================================

    st.markdown("## ⚠️ Measurement Rejection Details")

    rejected_df = filtered_df[
        filtered_df["status_normalized"] == "rejected"
    ]

    if not rejected_df.empty:

        rejection_display = rejected_df[
            ["timestamp", "driver_id", "reasons"]
        ].copy()

        rejection_display = rejection_display.rename(
            columns={
                "timestamp": "Timestamp",
                "driver_id": "Driver ID",
                "reasons": "Rejection Reason"
            }
        )

        st.dataframe(
            rejection_display,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.success(
            "No rejected measurements in the selected records."
        )

    # =========================================
    # FOOTER
    # =========================================

    st.markdown("---")

    st.html("""
    <div style="
        text-align:center;
        color:#8997AA;
        font-size:12px;
        padding:10px;
    ">
        PULSECAM FLEETGUARD · RESEARCH PROTOTYPE · 2026
        <br>
        For research and demonstration purposes only.
        Not a medical diagnostic or fitness-for-duty system.
    </div>
    """)
