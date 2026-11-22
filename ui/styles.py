"""
Startup-grade design system and CSS styling for ClaimGuard AI.
Styled to match the premium OLED dark palette:
- Main background: #080808
- Card background: #111114
- Secondary card background: #171215
- Elevated background: #1C1719
- Primary accent: #E63932
- Secondary accent: #FF554D
- Primary text: #FFFFFF
- Secondary text: #A1A1AA
- Border: #303035
- Success: #55C991
- Warning: #F2B94B
- Error: #F16B6B
"""

import base64
from pathlib import Path
import streamlit as st  # type: ignore

ASSETS_DIR = Path(__file__).parent / "assets"
BG_IMAGE_PATH = ASSETS_DIR / "background.png"


def _get_bg_base64() -> str:
    """Read the background image and convert to base64."""
    if BG_IMAGE_PATH.exists():
        try:
            return base64.b64encode(BG_IMAGE_PATH.read_bytes()).decode()
        except Exception:
            return ""
    return ""


def apply_styles():
    """Apply global dark theme with consistent charcoal cards, crimson accents, and zero white containers."""
    bg_b64 = _get_bg_base64()

    bg_css = ""
    if bg_b64:
        bg_css = f"""
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(135deg, rgba(8, 8, 8, 0.65) 0%, rgba(12, 12, 15, 0.85) 100%),
                        url("data:image/png;base64,{bg_b64}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}
        """

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Global Theme & Typography */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #FFFFFF !important;
            background-color: #080808 !important;
        }}

        code, pre, .mono {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* App Background */
        {bg_css}

        /* Streamlit Header & Toolbar */
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        [data-testid="stToolbar"] {{
            right: 1.5rem;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: #0C0C0E !important;
            border-right: 1px solid #303035 !important;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: #A1A1AA !important;
        }}

        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }}

        /* Top Hero Branding */
        .brand-hero {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 16px !important;
            padding: 24px 28px !important;
            margin-bottom: 24px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}

        .brand-hero:hover {{
            border-color: #E63932 !important;
            box-shadow: 0 8px 30px rgba(230, 57, 50, 0.15) !important;
        }}

        .brand-title {{
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(90deg, #FFFFFF 0%, #FFB4B4 50%, #FF554D 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            margin-bottom: 6px;
            letter-spacing: -0.02em;
        }}

        .brand-subtitle {{
            font-size: 14.5px;
            color: #A1A1AA !important;
            line-height: 1.5;
            font-weight: 400;
        }}

        /* Live Status Indicator */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        .status-online {{
            background-color: rgba(85, 201, 145, 0.15) !important;
            color: #55C991 !important;
            border: 1px solid rgba(85, 201, 145, 0.40) !important;
        }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #55C991;
            box-shadow: 0 0 0 0 rgba(85, 201, 145, 0.7);
            animation: pulse 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
        }}

        @keyframes pulse {{
            to {{
                box-shadow: 0 0 0 8px rgba(85, 201, 145, 0);
            }}
        }}

        /* Metric Cards */
        .metric-card,
        div[data-testid="stMetric"] {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 16px !important;
            padding: 18px 20px !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}

        .metric-card:hover,
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px) !important;
            border-color: #E63932 !important;
            box-shadow: 0 8px 24px rgba(230, 57, 50, 0.20) !important;
        }}

        .metric-label,
        div[data-testid="stMetricLabel"] p {{
            font-size: 12px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            color: #A1A1AA !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 6px !important;
        }}

        .metric-value,
        div[data-testid="stMetricValue"] div {{
            font-size: 26px !important;
            font-weight: 800 !important;
            color: #FFFFFF !important;
            line-height: 1.1 !important;
        }}

        .metric-sub,
        div[data-testid="stMetricDelta"] {{
            font-size: 12px !important;
            color: #55C991 !important;
            font-weight: 500 !important;
            margin-top: 4px !important;
        }}

        /* Workflow Cards */
        .workflow-card {{
            background: #171215 !important;
            border-radius: 14px !important;
            padding: 16px 12px !important;
            text-align: center !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }}

        .workflow-card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5) !important;
        }}

        /* Quick Action Cards */
        .quick-action-card {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 16px !important;
            padding: 20px !important;
            height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}

        .quick-action-card:hover {{
            transform: translateY(-2px) !important;
            border-color: #E63932 !important;
            box-shadow: 0 8px 24px rgba(230, 57, 50, 0.22) !important;
        }}

        /* Verdict Badges */
        .verdict-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: 8px;
            font-size: 12.5px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}

        .verdict-supported {{
            background-color: rgba(85, 201, 145, 0.15) !important;
            color: #55C991 !important;
            border: 1px solid rgba(85, 201, 145, 0.40) !important;
        }}

        .verdict-contradicted {{
            background-color: rgba(230, 57, 50, 0.15) !important;
            color: #F16B6B !important;
            border: 1px solid rgba(230, 57, 50, 0.45) !important;
        }}

        .verdict-insufficient {{
            background-color: rgba(242, 185, 75, 0.15) !important;
            color: #F2B94B !important;
            border: 1px solid rgba(242, 185, 75, 0.40) !important;
        }}

        /* Claim Cards */
        .claim-card-container {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 14px !important;
            padding: 18px 20px !important;
            margin-bottom: 12px !important;
            transition: border-color 0.2s ease !important;
        }}

        .claim-card-container:hover {{
            border-color: #4A4A52 !important;
        }}

        .claim-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }}

        .claim-id-pill {{
            background: #1C1719 !important;
            color: #FFFFFF !important;
            border: 1px solid #303035 !important;
            font-size: 11.5px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 6px;
            letter-spacing: 0.05em;
        }}

        .claim-text {{
            font-size: 15px;
            font-weight: 600;
            color: #FFFFFF !important;
            line-height: 1.5;
            margin-bottom: 10px;
        }}

        /* Evidence Box */
        .evidence-box {{
            background-color: #171215 !important;
            border: 1px solid #303035 !important;
            border-left: 3px solid #E63932 !important;
            border-radius: 0 10px 10px 0 !important;
            padding: 12px 16px !important;
            margin-top: 8px !important;
            font-size: 13.5px !important;
            color: #FFFFFF !important;
            line-height: 1.5 !important;
        }}

        .evidence-citation {{
            display: inline-block;
            font-size: 11.5px;
            font-weight: 600;
            color: #FFAAA8 !important;
            background: rgba(230, 57, 50, 0.20) !important;
            border: 1px solid rgba(230, 57, 50, 0.35) !important;
            padding: 2px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
        }}

        /* Streamlit Chat Messages */
        [data-testid="stChatMessage"] {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 16px !important;
            padding: 16px 20px !important;
            margin-bottom: 12px !important;
            color: #FFFFFF !important;
        }}

        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {{
            background: #171215 !important;
            border: 1px solid #303035 !important;
        }}

        /* Streamlit Expanders (Zero White Background) */
        [data-testid="stExpander"],
        details {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            margin-bottom: 12px !important;
        }}

        summary[data-testid="stExpanderSummary"],
        summary {{
            background: #111114 !important;
            color: #FFFFFF !important;
            padding: 12px 16px !important;
            border-radius: 14px !important;
            transition: color 0.15s ease !important;
        }}

        summary[data-testid="stExpanderSummary"]:hover,
        summary:hover {{
            color: #FF554D !important;
        }}

        [data-testid="stExpanderDetails"] {{
            background: #171215 !important;
            border-top: 1px solid #303035 !important;
            padding: 14px 18px !important;
            color: #A1A1AA !important;
        }}

        /* Streamlit Tabs (Settings & Benchmarks) */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            background: transparent !important;
            border-bottom: 1px solid #303035 !important;
            gap: 6px !important;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"] {{
            background: transparent !important;
            color: #A1A1AA !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            border: none !important;
            transition: color 0.15s ease, background-color 0.15s ease !important;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
            color: #FFFFFF !important;
            background: rgba(255, 255, 255, 0.04) !important;
        }}

        [data-testid="stTabs"] [aria-selected="true"] {{
            color: #FF554D !important;
            border-bottom: 2px solid #E63932 !important;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-border"] {{
            background-color: #E63932 !important;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-panel"] {{
            padding-top: 16px !important;
        }}

        /* Streamlit DataFrames and Tables */
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        div[data-testid="stDataEditor"] {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 14px !important;
            overflow: hidden !important;
        }}

        /* Streamlit File Uploader */
        [data-testid="stFileUploader"] {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 16px !important;
            padding: 14px 18px !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: #171215 !important;
            border: 1px dashed #303035 !important;
            border-radius: 12px !important;
            color: #A1A1AA !important;
            transition: border-color 0.2s ease, background 0.2s ease !important;
        }}

        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: #E63932 !important;
            background: #1C1719 !important;
        }}

        [data-testid="stFileUploaderDropzone"] * {{
            color: #A1A1AA !important;
        }}

        /* Streamlit Popovers & Dialogs */
        [data-testid="stPopoverBody"],
        div[data-baseweb="popover"],
        div[data-baseweb="menu"] {{
            background: #171215 !important;
            border: 1px solid #303035 !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7) !important;
            color: #FFFFFF !important;
        }}

        div[data-baseweb="menu"] ul {{
            background: #171215 !important;
        }}

        div[data-baseweb="menu"] li {{
            background: transparent !important;
            color: #A1A1AA !important;
        }}

        div[data-baseweb="menu"] li:hover {{
            background: rgba(230, 57, 50, 0.15) !important;
            color: #FFFFFF !important;
        }}

        [data-testid="stModal"],
        [data-testid="stDialog"] {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 16px !important;
        }}

        /* Streamlit Alerts / Notifications */
        div[data-testid="stAlert"] {{
            background: #171215 !important;
            border: 1px solid #303035 !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
        }}

        div[data-testid="stAlert"] p {{
            color: #A1A1AA !important;
        }}

        /* Streamlit Form Inputs: text_input, text_area, selectbox */
        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="base-input"] {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 10px !important;
            color: #FFFFFF !important;
        }}

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {{
            background: transparent !important;
            color: #FFFFFF !important;
        }}

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within {{
            border-color: #E63932 !important;
            box-shadow: 0 0 0 1px #E63932 !important;
        }}

        div[data-baseweb="select"] > div {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 10px !important;
            color: #FFFFFF !important;
        }}

        /* Outline Buttons */
        div.stButton > button,
        div.stDownloadButton > button,
        div.stFormSubmitButton > button,
        button[data-testid="baseButton-secondary"],
        .button-outline {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
            border-radius: 10px !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            letter-spacing: -0.01em !important;
            padding: 8px 16px !important;
            height: 38px !important;
            min-height: 38px !important;
            background-color: transparent !important;
            background: transparent !important;
            color: #FFFFFF !important;
            border: 1px solid #303035 !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1) !important;
            cursor: pointer !important;
        }}

        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        div.stFormSubmitButton > button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        .button-outline:hover {{
            background-color: rgba(230, 57, 50, 0.08) !important;
            background: rgba(230, 57, 50, 0.08) !important;
            color: #FFFFFF !important;
            border-color: #E63932 !important;
            box-shadow: 0 0 12px rgba(230, 57, 50, 0.22) !important;
            transform: translateY(-1px) !important;
        }}

        div.stButton > button:active,
        div.stDownloadButton > button:active,
        button[data-testid="baseButton-secondary"]:active {{
            transform: scale(0.98) !important;
        }}

        /* Primary Action Buttons */
        div.stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"],
        .button-outline-primary {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
            border-radius: 10px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            padding: 8px 20px !important;
            height: 40px !important;
            min-height: 40px !important;
            background-color: rgba(230, 57, 50, 0.15) !important;
            background: rgba(230, 57, 50, 0.15) !important;
            color: #FFFFFF !important;
            border: 1.5px solid #E63932 !important;
            box-shadow: 0 0 14px rgba(230, 57, 50, 0.25) !important;
            transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
            cursor: pointer !important;
        }}

        div.stButton > button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover,
        .button-outline-primary:hover {{
            background-color: #E63932 !important;
            background: #E63932 !important;
            color: #FFFFFF !important;
            border-color: #FF554D !important;
            box-shadow: 0 0 22px rgba(230, 57, 50, 0.65), 0 4px 12px rgba(0, 0, 0, 0.5) !important;
            transform: translateY(-1.5px) !important;
        }}

        /* shadcn Prompt Input Card */
        .shadcn-prompt-card {{
            background: #111114 !important;
            border: 1px solid #303035 !important;
            border-radius: 16px !important;
            padding: 16px 18px 12px 18px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
            margin-bottom: 20px !important;
        }}

        .shadcn-prompt-card:focus-within {{
            border-color: #E63932 !important;
            box-shadow: 0 10px 32px rgba(230, 57, 50, 0.22) !important;
        }}

        .shadcn-attachment-chip {{
            display: inline-flex !important;
            align-items: center !important;
            gap: 7px !important;
            background: #171215 !important;
            border: 1px solid #303035 !important;
            border-radius: 8px !important;
            padding: 4px 10px !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            color: #FFFFFF !important;
        }}

        div[data-testid="stTextArea"].shadcn-prompt-input textarea {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 4px 2px 8px 2px !important;
            font-size: 15px !important;
            line-height: 1.5 !important;
            color: #FFFFFF !important;
            resize: none !important;
        }}

        div[data-testid="stTextArea"].shadcn-prompt-input textarea:focus {{
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
        }}

        div.shadcn-send-btn > div.stButton > button {{
            width: 38px !important;
            height: 38px !important;
            min-height: 38px !important;
            border-radius: 9999px !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: #E63932 !important;
            background: #E63932 !important;
            border: 1px solid #FF554D !important;
            color: #FFFFFF !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            box-shadow: 0 0 14px rgba(230, 57, 50, 0.45) !important;
            transition: all 0.15s ease !important;
        }}

        div.shadcn-send-btn > div.stButton > button:hover {{
            background-color: #FF554D !important;
            background: #FF554D !important;
            box-shadow: 0 0 22px rgba(230, 57, 50, 0.75) !important;
            transform: scale(1.06) !important;
        }}

        div.shadcn-plus-btn > div.stButton > button {{
            width: 36px !important;
            height: 36px !important;
            min-height: 36px !important;
            border-radius: 9999px !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 17px !important;
            border: 1px solid #303035 !important;
        }}

        .shadcn-prompt-card div[data-testid="stSelectbox"] > div > div {{
            background: #171215 !important;
            border: 1px solid #303035 !important;
            border-radius: 8px !important;
            min-height: 36px !important;
            height: 36px !important;
            padding: 0 8px !important;
            font-size: 12.5px !important;
            color: #FFFFFF !important;
        }}

        .shadcn-prompt-card div[data-testid="stSelectbox"] > div > div:hover {{
            border-color: #E63932 !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
