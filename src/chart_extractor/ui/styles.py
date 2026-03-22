from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            font-family: 'Inter', sans-serif;
            color: #0F172A !important;
        }
        p, div, label, li, span, h1, h2, h3, h4, h5, h6 { color: #0F172A !important; }
        [data-testid="stAppViewContainer"] { background: #FFFFFF; }
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }
        h1 {
            background: linear-gradient(to right, #2563EB, #4F46E5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent !important;
        }
        div.stButton > button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        }
        .stDataFrame {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
        }
        [data-testid="stAlert"] {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0;
        }
        input { color: #0F172A !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

