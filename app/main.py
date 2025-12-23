"""Reservoir Engine - Main Streamlit Application.

A Databricks-native reservoir engineering application for:
- Exploratory Data Analysis (EDA)
- Decline Curve Analysis (DCA)
- Type Curve Generation
- Uplift/ML Analysis
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from app.config import load_config
from app.database import init_backend, get_backend
from app.pages import eda, dca, type_curves, uplift


# Page configuration
st.set_page_config(
    page_title="Reservoir Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_databricks_theme():
    """Apply Databricks branding."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif !important;
            color: #0B2026 !important;
        }
        
        .main { background-color: #F9F7F4; }
        [data-testid="stSidebar"] { background-color: #EEEDE9; }
        
        h1, h2, h3, h4, h5, h6 {
            color: #0B2026 !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 700 !important;
        }
        
        .stButton > button {
            background-color: #FF3621;
            color: white !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500;
            border: none;
            border-radius: 4px;
        }
        
        .stButton > button:hover {
            background-color: #E62E1C;
            color: white !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #0B2026 !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500;
            font-size: 1.1rem !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #FF3621 !important;
            border-bottom-color: #FF3621 !important;
            font-weight: 700 !important;
        }
        
        /* Metric styling */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
        }
        
        [data-testid="stMetricValue"] {
            color: #FF3621 !important;
        }
        
        /* Hide default Streamlit elements above sidebar content */
        [data-testid="stSidebarHeader"] {
            display: none !important;
        }
        
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
        
        /* Hide default page navigation in sidebar */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        section[data-testid="stSidebar"] > div:first-child > div:first-child {
            padding-top: 1rem !important;
        }
        
        /* Hide the hamburger menu and decorations */
        #MainMenu {
            visibility: hidden;
        }
        
        header[data-testid="stHeader"] {
            display: none;
        }
        
        /* Remove extra top padding */
        .block-container {
            padding-top: 1rem !important;
        }
        
        /* Sidebar title - large and prominent */
        .sidebar-title {
            font-size: 30px;
            font-weight: 700;
            color: #0B2026;
            margin: 1rem 0 1.5rem 0;
            line-height: 1.1;
        }
        
        /* Navigation buttons */
        [data-testid="stSidebar"] .stButton > button {
            background-color: transparent;
            color: #0B2026 !important;
            border: 1px solid #D0D0D0;
            text-align: left;
            justify-content: flex-start;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #E0E0E0;
            border-color: #FF3621;
        }
        
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background-color: #FF3621;
            color: white !important;
            border: none;
        }
        
        /* Status indicators */
        .status-connected {
            color: #00A972;
        }
        
        .status-disconnected {
            color: #E62E1C;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "backend_initialized" not in st.session_state:
        init_backend(st.session_state.config)
        st.session_state.backend_initialized = True

    if "current_page" not in st.session_state:
        st.session_state.current_page = "EDA"

    if "selected_uwis" not in st.session_state:
        st.session_state.selected_uwis = []


def render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        # Logo
        logo_path = project_root / "assets" / "databricks_logo.svg"
        if logo_path.exists():
            st.image(str(logo_path), width=150)
        
        # Title
        st.markdown('<p class="sidebar-title">Reservoir Engine</p>', unsafe_allow_html=True)

        st.divider()

        # Navigation
        st.markdown("**Navigation**")

        pages = {
            "EDA": "Exploratory Data Analysis",
            "DCA": "Decline Curve Analysis",
            "Type Curves": "Type Curve Generation",
            "Uplift": "ML Uplift Analysis",
        }

        for page_key, description in pages.items():
            is_active = st.session_state.current_page == page_key
            button_style = "primary" if is_active else "secondary"

            if st.button(
                page_key,
                key=f"nav_{page_key}",
                use_container_width=True,
                type=button_style,
            ):
                st.session_state.current_page = page_key
                st.rerun()

            st.caption(description)

        st.divider()

        # Status - simple text only
        st.markdown("**Status**")
        config = st.session_state.config
        backend = get_backend()

        mode_label = "TEST" if config.is_test_mode else "PROD"
        connected = "Connected" if backend.is_connected() else "Disconnected"
        st.caption(f"Mode: {mode_label} | Data: {connected}")

        # Well stats - simple text
        try:
            wells = backend.get_well_header()
            total = len(wells)
            hz = (wells["hole_direction"] == "HORIZONTAL").sum() if "hole_direction" in wells.columns else 0
            st.caption(f"Wells: {total:,} total, {hz:,} horizontal")
        except Exception:
            st.caption("Error loading wells")

        # Selected wells count - always show
        n_selected = len(st.session_state.selected_uwis)
        st.caption(f"Selected: {n_selected:,} wells")


def render_main_content():
    """Render the main content based on current page."""
    page = st.session_state.current_page

    if page == "EDA":
        eda.render()
    elif page == "DCA":
        dca.render()
    elif page == "Type Curves":
        type_curves.render()
    elif page == "Uplift":
        uplift.render()


def main():
    """Main application entry point."""
    apply_databricks_theme()
    initialize_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
