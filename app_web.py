import base64
import csv
from datetime import datetime
import os
import time
from crewai import Agent, Crew, Process, Task, LLM
from fpdf import FPDF
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURAÇÃO DA MARCA E PÁGINA
# ==========================================
st.set_page_config(
    page_title="NOVUS AI | Inteligência de Negócios",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. DESIGN SYSTEM: ÍCONES SVG TWO-TONE
# ==========================================
ICO_CHART = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px; margin-bottom: 2px;"><rect x="10" y="10" width="10" height="10" fill="#FF8A00" stroke="none" opacity="0.8"/><rect x="3" y="4" width="14" height="14" rx="2"/><path d="M7 18V10M11 18V6"/></svg>'
ICO_PROCESS = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px; margin-bottom: 2px;"><circle cx="16" cy="16" r="6" fill="#FF007A" stroke="none" opacity="0.8"/><path d="M12 3L5 13h7l-2 8 9-11h-7z"/></svg>'
ICO_GIFT = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px; margin-bottom: 2px;"><rect x="12" y="12" width="10" height="10" fill="#FF8A00" stroke="none" opacity="0.8"/><polyline points="20 12 20 22 4 22 4 12"/><rect x="2" y="7" width="20" height="5"/><line x1="12" y1="22" x2="12" y2="7"/><path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z"/><path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z"/></svg>'
ICO_BOT = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px; margin-bottom: 2px;"><circle cx="16" cy="16" r="6" fill="#FF007A" stroke="none" opacity="0.8"/><rect x="3" y="7" width="14" height="10" rx="2"/><path d="M7 7V3M13 7V3M10 11v2M7 15h6"/></svg>'
ICO_LIGHTNING = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px; margin-bottom: 2px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" fill="#FF8A00" stroke="none" opacity="0.8"/><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
ICO_LOCK = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px; margin-bottom: 2px;"><rect x="12" y="14" width="10" height="8" fill="#FF8A00" stroke="none" opacity="0.8"/><rect x="4" y="10" width="14" height="10" rx="2" ry="2"/><path d="M7 10V7a5 5 0 0 1 10 0v3"/></svg>'
ICO_CHECK = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><circle cx="16" cy="16" r="7" fill="#FF8A00" stroke="none" opacity="0.8"/><polyline points="20 6 9 17 4 12"/></svg>'
ICO_SHIELD = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M15 22s8-4 8-10V5l-8-3-8 3v7c0 6-8 10-8 10z" fill="#FF007A" stroke="none" opacity="0.8"/><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
ICO_STAR = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><polygon points="15 5 18 12 25 12 19 17 21 24 15 20 9 24 11 17 5 12 12 12" fill="#FF8A00" stroke="none" opacity="0.8"/><polygon points="12 2 15 9 22 9 16 14 18 21 12 17 6 21 8 14 2 9 9"/></svg>'
ICO_CURSOR = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FF8A00" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M13 13l6-6-15-4L8 18l3-4 4 7 3-2-4-7z"/></svg>'
ICO_ROCKET = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FF8A00" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 4px; margin-bottom: 2px;"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2l.5-.5a2.5 2.5 0 0 0 1.96-3.96l5-5c1.26-1.5 2-5 2-5s-3.74.5-5 2l-5 5a2.5 2.5 0 0 0-3.96 1.96l-.5.5z"/></svg>'
ICO_TARGET = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FF8A00" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 4px; margin-bottom: 2px;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
ICO_LOCK_SM = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FF8A00" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 4px; margin-bottom: 2px;"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>'

# ==========================================
# 3. INJEÇÃO DE CSS: ESTILOS E COMPONENTES
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800;900&display=swap');

    html, body, [class*="css"], p, h1, h2, h3, h4, h5, h6, span, div, button { font-family: 'Montserrat', sans-serif !important; }
    .st-icon, [data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
    .stApp { background-color: #0B0B0F; color: #E2E8F0; }
    .gradient-text { background: linear-gradient(90deg, #FF007A 0%, #FF8A00 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} 
    header, header[data-testid="stHeader"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebar"] { background-color: #050508 !important; border-right: 1px solid #16161D !important; }
    
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
    
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .stTabs [data-baseweb="tab-panel"] { animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; }

    .mobile-logo-container { display: none; }
    @media (max-width: 768px) {
        .block-container { padding-top: 0.5rem !important; }
        .mobile-logo-container { display: block; text-align: center; margin-bottom: 0px; padding-top: 0px; padding-bottom: 10px; border-bottom: 1px solid #1E1E26; }
        .infographic-flow { flex-direction: column; align-items: stretch; } 
        .flow-arrow { transform: rotate(90deg); margin: 5px 0; font-size: 20px; text-align: center;} 
    }

    .infographic-flow { display: flex; align-items: center; justify-content: space-between; gap: 15px; margin-bottom: 30px; }
    .flow-arrow { color: #2E2E38; font-size: 28px; font-weight: 900; display: flex; align-items: center; justify-content: center; transition: color 0.3s ease; }
    .infographic-flow:hover .flow-arrow { color: #FF8A00; opacity: 0.5; }
    
    .info-step-card, .feature-card {
        background-color: #13131A; border: 1px solid #1E1E26; border-radius: 12px; padding: 24px;
        position: relative; overflow: hidden; height: 140px; 
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); display: flex; flex-direction: column; justify-content: space-between; flex: 1;
    }
    
    .step-1 { border-color: rgba(255, 138, 0, 0.3); box-shadow: 0 0 15px rgba(255, 138, 0, 0.05); }
    .info-step-card:hover { height: 380px; border-color: #FF8A00; box-shadow: 0 12px 30px rgba(255, 138, 0, 0.15); transform: translateY(-4px); z-index: 10; }
    .feature-card:hover { height: 260px; border-color: #FF8A00; box-shadow: 0 12px 30px rgba(255, 138, 0, 0.15); transform: translateY(-4px); }

    .info-step-number { position: absolute; top: -10px; right: 15px; font-size: 70px; font-weight: 900; background: linear-gradient(180deg, rgba(255,138,0,0.15) 0%, rgba(255,0,122,0.0) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; z-index: 0; user-select: none; }
    .info-step-content, .feature-content { position: relative; z-index: 1; }
    .info-step-title, .feature-title { color: #FFFFFF; font-weight: 800; font-size: 18px; margin-bottom: 6px; display: flex; align-items: center; }
    
    .info-step-desc, .feature-desc { max-height: 0; opacity: 0; overflow: hidden; transition: all 0.4s ease-in-out; color: #94A3B8; font-size: 13px; line-height: 1.5; }
    .info-step-card:hover .info-step-desc { max-height: 300px; opacity: 1; margin-top: 12px; }
    .feature-card:hover .feature-desc { max-height: 160px; opacity: 1; margin-top: 12px; }

    @keyframes pulse-text { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
    .hover-hint { color: #FF8A00; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; animation: pulse-text 2s infinite; transition: opacity 0.3s; margin-top: auto; display: flex; align-items: center; gap: 5px; }
    .info-step-card:hover .hover-hint { opacity: 0; max-height: 0; visibility: hidden; }

    .info-step-stat, .feature-stat { background-color: rgba(255, 138, 0, 0.1); border: 1px solid rgba(255, 138, 0, 0.2); color: #FF8A00; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 800; width: fit-content; position: relative; z-index: 1; opacity: 0; transition: opacity 0.3s ease; display: none; align-items: center; }
    .info-step-card:hover .info-step-stat, .feature-card:hover .feature-stat { opacity: 1; display: flex; margin-top: 10px; }

    /* UPLOADER E BOTÕES */
    [data-testid="stFileUploader"] { background-color: #13131A; border: 1px solid #1E1E26; border-radius: 10px; padding: 12px; }
    [data-testid="stFileUploader"] section { padding: 16px !important; background-color: transparent !important; border: 2px dashed #2E2E38 !important; border-radius: 8px !important; transition: all 0.3s ease; }
    [data-testid="stFileUploader"] section:hover { border-color: #FF8A00 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    [data-testid="stFileUploader"] section small { color: #64748B !important; }
    
    [data-testid="stFileUploader"] button { background: #1A1A24 !important; color: transparent !important; border: 1px solid #2E2E38 !important; border-radius: 6px !important; transition: all 0.3s ease; position: relative; }
    [data-testid="stFileUploader"] button::after { content: "Procurar Arquivo"; position: absolute; left: 0; right: 0; top: 0; bottom: 0; display: flex; align-items: center; justify-content: center; margin-left: 20px; color: #FFFFFF !important; font-weight: 600 !important; font-size: 14px; transition: all 0.3s ease; }
    [data-testid="stFileUploader"] button::before { content: ""; position: absolute; left: 50%; top: 50%; transform: translate(-75px, -50%); width: 16px; height: 16px; z-index: 1; transition: all 0.3s ease; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E"); background-size: contain; background-repeat: no-repeat; }
    [data-testid="stFileUploader"] button:hover { border-color: #FF8A00 !important; box-shadow: 0 4px 12px rgba(255, 138, 0, 0.1) !important; }
    [data-testid="stFileUploader"] button:hover::after { color: #FF8A00 !important; }
    [data-testid="stFileUploader"] button:hover::before { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FF8A00' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E"); }

    [data-testid="stDownloadButton"] button p { display: flex; align-items: center; justify-content: center; }
    [data-testid="stDownloadButton"] button p::before { content: ""; display: inline-block; width: 16px; height: 16px; margin-right: 8px; vertical-align: middle; transition: all 0.3s ease; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='7 10 12 15 17 10'/%3E%3Cline x1='12' y1='15' x2='12' y2='3'/%3E%3C/svg%3E"); background-size: contain; background-repeat: no-repeat; }
    [data-testid="stDownloadButton"] button:hover p::before { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FF8A00' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='7 10 12 15 17 10'/%3E%3Cline x1='12' y1='15' x2='12' y2='3'/%3E%3C/svg%3E"); }

    .stTabs [data-baseweb="tab-list"] { gap: 32px; background-color: transparent; border-bottom: 1px solid #1E1E26; }
    .stTabs [data-baseweb="tab"] { color: #64748B; font-weight: 600; padding: 16px 0; }
    .stTabs [aria-selected="true"] { color: #FF8A00 !important; border-bottom-color: #FF8A00 !important; }

    .stButton > button, .stLinkButton > a { background: linear-gradient(90deg, #FF007A 0%, #FF8A00 100%) !important; color: #FFFFFF !important; border-radius: 8px !important; border: none !important; padding: 12px 24px !important; font-weight: 800 !important; width: 100% !important; box-shadow: 0px 4px 15px rgba(255, 138, 0, 0.25) !important; }
    .btn-secundario > button { background: #1A1A24 !important; border: 1px solid #2E2E38 !important; box-shadow: none !important; }
    .btn-secundario > button:hover { border-color: #FF8A00 !important; color: #FF8A00 !important; }

    .badge { background: #13131A; border: 1px solid #1E1E26; padding: 10px 12px; border-radius: 8px; font-size: 11px; color: #94A3B8; display: flex; align-items: center; transition: all 0.3s ease; cursor: pointer; }
    .badge:hover { border-color: #FF8A00; color: #FFFFFF; background: #1A1A24; transform: translateY(-4px); box-shadow: 0 4px 12px rgba(255, 138, 0, 0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 4. CARREGAMENTO DE IMAGENS E CLASSE PDF 
# ==========================================
def carregar_imagem_base64(caminho):
    try:
        with open(caminho, "rb") as f: return base64.b64encode(f.read()).decode()
    except: return None

nome_logo = "novus.gif"
logo_b64 = carregar_imagem_base64(nome_logo)

# ==========================================
# 5. SPLASH SCREEN (ANIMAÇÃO DE CARREGAMENTO INICIAL)
# ==========================================
if "splash_exibido" not in st.session_state:
    st.session_state["splash_exibido"] = False

if not st.session_state["splash_exibido"]:
    placeholder_splash = st.empty()
    with placeholder_splash.container():
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh; text-align: center;">
                <img src="data:image/gif;base64,{logo_b64}" width="220" style="margin-bottom: 25px; animation: fadeIn 1s ease-in-out;">
                <p style="color: #94A3B8; font-size: 14px; letter-spacing: 2px; text-transform: uppercase;">Carregando Inteligência Estratégica...</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    time.sleep(2.0)
    st.session_state["splash_exibido"] = True
    placeholder_splash.empty()
    st.rerun()

def salvar_lead(nome, email, whatsapp):
    arquivo = "leads_novus.csv"
    existe = os.path.exists(arquivo)
    with open(arquivo, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["Data/Hora", "Nome", "E-mail", "WhatsApp"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), nome, email, whatsapp])

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 18)
        self.set_text_color(255, 138, 0)
        self.cell(0, 10, "NOVUS AI - AUDITORIA EXECUTIVA", 0, 1, "C")
        self.set_draw_color(30, 30, 38)
        self.line(10, 25, 200, 25)
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Pagina {self.page_no()} | Processado por Inteligencia Artificial Autonoma - NOVUS AI", 0, 0, "C")

    def chapter_title(self, title):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(11, 11, 15)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(2)

    def chapter_body(self, body):
        self.set_font("helvetica", "", 10)
        self.set_text_color(51, 65, 85)
        body = body.replace("**", "").replace("*", "-")
        self.multi_cell(0, 6, body)
        self.ln()

df_exemplo = pd.DataFrame({
    "Produto": ["Licença Software SaaS", "Consultoria Estratégica VIP", "Mentoria Gravada (Online)", "Suporte Técnico Mensal", "Setup Manual de Sistemas", "E-book Guia de Vendas", "Implantação de E-commerce"],
    "Quantidade": [150, 8, 310, 200, 42, 850, 5],
    "Receita Total": [145000.00, 80000.00, 92000.00, 50000.00, 42000.00, 40000.00, 25000.00],
    "Custo Total": [30000.00, 20000.00, 15000.00, 45000.00, 38000.00, 8000.00, 28000.00]
})
csv_exemplo = df_exemplo.to_csv(index=False).encode('utf-8')


# ==========================================
# 6. BARRA LATERAL (SIDEBAR LIMPA)
# ==========================================
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

with st.sidebar:
    if logo_b64:
        st.markdown(f'<div style="text-align: center; margin-bottom: 5px;"><img src="data:image/gif;base64,{logo_b64}" width="180"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<h1 style="font-size: 32px; text-align: center; margin-bottom: 0px;"><span style="color: #FF007A;"><span style="font-weight: 900;">NOVUS</span> <span style="font-weight: 300;">AI</span></span></h1>', unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 13px; margin-bottom: 20px;'>Inteligência Estratégica</p>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='background-color: #13131A; border: 1px solid #1E1E26; border-radius: 8px; padding: 16px; margin-bottom: 20px;'>"
        f"<div style='color: #FF8A00; font-weight: 800; margin-bottom: 6px; font-size: 13px; display: flex; align-items: center;'>{ICO_LOCK} Privacidade 100%</div>"
        "<div style='color: #94A3B8; font-size: 12px; line-height: 1.4;'>Processamento neural seguro.</div>"
        "</div>", 
        unsafe_allow_html=True
    )


# ==========================================
# 7. HEADER MOBILE E ABAS DO SISTEMA
# ==========================================

if logo_b64:
    st.markdown(f'<div class="mobile-logo-container"><img src="data:image/gif;base64,{logo_b64}" width="160"></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="mobile-logo-container"><h1 style="font-size: 32px; margin-bottom: 0px;"><span style="color: #FF007A;"><span style="font-weight: 900;">NOVUS</span> <span style="font-weight: 300;">AI</span></span></h1></div>', unsafe_allow_html=True)


aba_auditoria, aba_sobre = st.tabs(["Auditoria Inteligente", "Sobre a Plataforma"])

with aba_sobre:
    st.markdown('<h1 style="font-size: 40px; margin-bottom: 10px;">Bem-vindo à <span class="gradient-text"><span style="font-weight: 900;">NOVUS</span> <span style="font-weight: 300;">AI</span></span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 16px; margin-bottom: 35px;">Acreditamos que toda empresa possui um <b>lucro oculto</b> mascarado em montanhas de dados complexos.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="feature-card"><div class="feature-content"><div class="feature-title">{ICO_BOT} IA Autônoma</div><div class="feature-desc">Nossos agentes analisam padrões profundos de compra, giro de estoque e margens de lucro sem intervenção humana, garantindo precisão absoluta nas decisões.</div></div><div class="feature-stat">{ICO_ROCKET} Alta Precisão</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="feature-card"><div class="feature-content"><div class="feature-title">{ICO_LIGHTNING} Ação Imediata</div><div class="feature-desc">Esqueça relatórios estáticos de 50 páginas. Entregamos um Plano de Ação executivo focado estritamente em marketing, conversão e otimização financeira.</div></div><div class="feature-stat">{ICO_TARGET} Foco em Retorno</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="feature-card"><div class="feature-content"><div class="feature-title">{ICO_LOCK} Segurança Local</div><div class="feature-desc">Arquitetura avançada operando offline ou via nuvem segura. Suas planilhas, custos e dados estratégicos protegidos.</div></div><div class="feature-stat">{ICO_LOCK_SM} 100% Confidencial</div></div>', unsafe_allow_html=True)

with aba_auditoria:
    st.markdown('<h1 class="gradient-text" style="font-weight: 900; font-size: 38px;">Descubra o seu Lucro Oculto</h1>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 15px; margin-bottom: 30px;'>Tenha o relatório completo que sua empresa precisa em apenas 1 clique através de inteligência artificial.</p>", unsafe_allow_html=True)

    html_infografico = (
        f'<div class="infographic-flow">'
        f'<div class="info-step-card step-1"><div class="info-step-number">1</div><div class="info-step-content"><div class="info-step-title">{ICO_CHART} A Planilha</div><div class="info-step-desc">Para a IA auditar o seu negócio, envie um <b>CSV</b> contendo 4 colunas básicas:<br><br>• <b>Produto:</b> Nome do item.<br>• <b>Quantidade:</b> Volume de vendas.<br>• <b>Receita Total:</b> Valor bruto faturado.<br>• <b>Custo Total:</b> Custo de produção/entrega.<br><br><i>O lucro oculto é encontrado cruzando custos vs. receitas.</i></div></div><div class="hover-hint">{ICO_CURSOR} Passe o mouse</div><div class="info-step-stat">4 Colunas Básicas</div></div>'
        f'<div class="flow-arrow">➔</div>'
        f'<div class="info-step-card"><div class="info-step-number">2</div><div class="info-step-content"><div class="info-step-title">{ICO_PROCESS} O Processo</div><div class="info-step-desc">Fluxo autônomo e imediato:<br><br>• <b>Análise:</b> Identifica produtos que estão sugando suas margens vs. produtos estrela.<br>• <b>Estratégia:</b> Criação de plano tático.<br>• <b>Geração:</b> PDF executivo automatizado.</div></div><div class="hover-hint">{ICO_CURSOR} Passe o mouse</div><div class="info-step-stat">100% Neural</div></div>'
        f'<div class="flow-arrow">➔</div>'
        f'<div class="info-step-card"><div class="info-step-number">3</div><div class="info-step-content"><div class="info-step-title">{ICO_GIFT} Benefícios</div><div class="info-step-desc">Diagnóstico em segundos:<br><br>• <b>Lucro Oculto:</b> Enxergue a margem real de cada item.<br>• <b>Plano de Ação:</b> O que cortar e o que escalar hoje.<br>• <b>Consultoria Premium:</b> Material comercializável com gráficos.</div></div><div class="hover-hint">{ICO_CURSOR} Passe o mouse</div><div class="info-step-stat">Ação Imediata</div></div>'
        f'</div>'
    )
    st.markdown(html_infografico, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1E1E26; margin-bottom: 30px;'><br>", unsafe_allow_html=True)

    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        arquivo_cliente = st.file_uploader("Selecione sua planilha de vendas (.csv)", type=["csv"])
    with col_up2:
        st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='btn-secundario'>", unsafe_allow_html=True)
        st.download_button(label="Baixar CSV de Exemplo", data=csv_exemplo, file_name="exemplo_vendas.csv", mime="text/csv", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if arquivo_cliente is not None:
        tabela = pd.read_csv(arquivo_cliente)
        
        if "Receita_Total" in tabela.columns: tabela = tabela.rename(columns={"Receita_Total": "Receita Total"})
        if "Custo_Total" in tabela.columns: tabela = tabela.rename(columns={"Custo_Total": "Custo Total"})
        
        if "Custo Total" in tabela.columns and "Receita Total" in tabela.columns:
            tabela['Lucro Líquido'] = tabela['Receita Total'] - tabela['Custo Total']
            tabela['Margem (%)'] = (tabela['Lucro Líquido'] / tabela['Receita Total']) * 100
            tabela = tabela.sort_values(by='Lucro Líquido', ascending=False).reset_index(drop=True)

        st.markdown("<br><h3 style='font-weight: 800;'>Visão Geral Financeira (Receita vs. Custos)</h3>", unsafe_allow_html=True)
        
        if "Custo Total" in tabela.columns and "Receita Total" in tabela.columns:
            st.bar_chart(data=tabela, x="Produto", y=["Receita Total", "Custo Total"], color=["#FF8A00", "#FF007A"])
        else:
            st.bar_chart(data=tabela, x="Produto", y="Receita Total", color="#FF8A00")

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-weight: 800; font-size: 20px;'>Identificação do Executivo</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 13px;'>Preencha seus dados para liberar o processamento neural da sua auditoria:</p>", unsafe_allow_html=True)
        
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            nome_lead = st.text_input("Seu Nome Completo", placeholder="Ex: Carlos Silva")
        with col_l2:
            email_lead = st.text_input("Seu E-mail Corporativo", placeholder="Ex: carlos@empresa.com")
        with col_l3:
            whats_lead = st.text_input("Seu WhatsApp", placeholder="Ex: (11) 99999-9999")

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Iniciar Processamento Neural", use_container_width=True):
            if not nome_lead or not email_lead or not whats_lead:
                st.warning("⚠️ Por favor, preencha todos os campos de contato (Nome, E-mail e WhatsApp) para liberar a auditoria.")
            else:
                salvar_lead(nome_lead, email_lead, whats_lead)
                
                grafico_temp = "grafico_pdf_temp.png"
                if "Custo Total" in tabela.columns and "Receita Total" in tabela.columns:
                    fig, ax1 = plt.subplots(figsize=(12, 7))
                    ax2 = ax1.twinx()

                    x = np.arange(len(tabela['Produto']))
                    width = 0.35

                    bar1 = ax1.bar(x - width/2, tabela['Receita Total'], width, label='Receita Bruta', color='#FF8A00', edgecolor='white', linewidth=1)
                    bar2 = ax1.bar(x + width/2, tabela['Custo Total'], width, label='Custo Total', color='#FF007A', edgecolor='white', linewidth=1)

                    line1 = ax2.plot(x, tabela['Margem (%)'], color='#22C55E', marker='o', linewidth=2.5, markersize=8, label='Margem de Lucro (%)')

                    ax1.set_ylabel('Valor Financeiro (R$)', fontweight='bold', color='#334155')
                    ax2.set_ylabel('Margem de Lucro (%)', fontweight='bold', color='#22C55E')
                    ax1.set_title('Auditoria de Rentabilidade: Ordem de Lucratividade', fontsize=16, fontweight='900', color='#0F172A', pad=15)
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(tabela['Produto'], rotation=45, ha='right', fontsize=9, fontweight='600')
                    ax1.grid(axis='y', linestyle='--', alpha=0.3)

                    def autolabel(rects, ax):
                        for rect in rects:
                            height = rect.get_height()
                            ax.annotate(f'R${height/1000:.0f}k',
                                        xy=(rect.get_x() + rect.get_width() / 2, height),
                                        xytext=(0, 3), textcoords="offset points",
                                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#334155')
                    autolabel(bar1, ax1)
                    autolabel(bar2, ax1)

                    lines_1, labels_1 = ax1.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, shadow=True)
                    
                    plt.tight_layout()
                    fig.savefig(grafico_temp, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
                    plt.close(fig)

                with st.status("**Inicializando Rede Neural Executiva...**", expanded=True) as status:
                    
                    if os.environ.get("GROQ_API_KEY"):
                        st.write("Conectando à API da Groq (Nuvem)...")
                        chave_groq = os.environ.get("GROQ_API_KEY")
                        # MODELO ATUALIZADO AQUI
                        modelo_local = LLM(model="groq/llama-3.1-8b-instant", api_key=chave_groq)
                    else:
                        st.write("Conectando ao LLM local (Ollama / Llama 3)...")
                        modelo_local = LLM(model="ollama/llama3", base_url="http://localhost:11434")

                    st.write("Acordando Agente Analista Financeiro...")
                    instrucao_mestre = (
                        "ATENÇÃO MÁXIMA: VOCÊ DEVE RESPONDER INTEGRALMENTE EM PORTUGUÊS DO BRASIL. "
                        "É PROIBIDO escrever qualquer palavra, frase ou introdução em inglês. "
                        "Atue como Consultor Sênior de uma Big Four. Use gramática impecável, ortografia perfeita e jargões financeiros em português "
                        "(OPEX, ROI, LTV, Cross-sell, Margem de Contribuição, CAC)."
                    )
                    
                    analista = Agent(role="Head de Dados e Auditoria", goal="Extrair KPIs e classificar a rentabilidade.", backstory=instrucao_mestre, llm=modelo_local)
                    consultor = Agent(role="Estrategista C-Level", goal="Gerar plano de ação executivo com base nos dados.", backstory=instrucao_mestre, llm=modelo_local)

                    st.write("Processando cruzamento avançado de margens...")
                    
                    # LIMITE DE DADOS (HEAD 30) ATUALIZADO AQUI
                    dados_texto = tabela.head(30).to_string()
                    
                    prompt_analista = f"EM PORTUGUÊS DO BRASIL: Analise esta base de dados rigorosamente: {dados_texto}. 1. Apresente os KPIs de Lucro e Margem % em formato de lista (bullet points). 2. Classifique o portfólio usando a Matriz BCG adaptada. Retorne um diagnóstico profundo."
                    t1 = Task(description=prompt_analista, expected_output="Diagnóstico financeiro em português.", agent=analista)
                    
                    prompt_consultor = "EM PORTUGUÊS DO BRASIL: Com base no diagnóstico do Analista, redija o 'Executive Summary & Plano de Ação Tático' para a Diretoria/CEO. Divida o texto obrigatoriamente em 3 Pilares: Pilar 1: Tração e Escala, Pilar 2: Reestruturação (Turnaround) de Prejuízos, Pilar 3: Otimização de Portfólio. Especifique os valores reais em R$."
                    t2 = Task(description=prompt_consultor, expected_output="Plano tático C-Level estruturado em 3 pilares totalmente em português.", agent=consultor, context=[t1])

                    equipe = Crew(agents=[analista, consultor], tasks=[t1, t2], process=Process.sequential)
                    
                    st.write("Redigindo relatório executivo final...")
                    resultado = equipe.kickoff()

                    pdf = PDF()
                    pdf.add_page()
                    pdf.chapter_title("1. SUMARIO EXECUTIVO & ESTRATEGIA C-LEVEL")
                    
                    texto_limpo = str(resultado).encode('latin-1', 'replace').decode('latin-1')
                    pdf.chapter_body(texto_limpo)
                    
                    if os.path.exists(grafico_temp):
                        pdf.add_page()
                        pdf.chapter_title("2. MATRIZ FINANCEIRA E MAPEAMENTO DE GARGALOS")
                        pdf.chapter_body("O painel analitico abaixo cruza a Receita Bruta, o Custo Total e a Linha de Tendencia da Margem de Lucro (%). Produtos ordenados automaticamente da maior para a menor lucratividade, facilitando a identificacao de ativos 'Estrela' e gargalos operacionais ('Abacaxis').")
                        pdf.ln(5)
                        pdf.image(grafico_temp, x=10, w=190)
                        os.remove(grafico_temp)
                    
                    pdf.output("NOVUS_AI_Estrategia.pdf")
                    
                    status.update(label="**Auditoria Concluída com Sucesso!**", state="complete", expanded=False)

                st.markdown(f"""
                <div style="position: relative; margin-top: 30px; margin-bottom: 20px; border-radius: 12px; overflow: hidden; border: 1px solid #1E1E26;">
                    <div style="filter: blur(6px); opacity: 0.4; background: linear-gradient(180deg, #13131A 0%, #050508 100%); height: 180px; display: flex; align-items: center; justify-content: center; padding: 20px; font-family: monospace; color: #94A3B8; user-select: none; line-height: 1.6;">
                        [DADOS CONFIDENCIAIS OCULTOS]<br><br>
                        Estratégia traçada para otimização de margens de lucro com base nos dados cruzados.<br>
                        Foi detectado que o produto [X] consome 80% da sua margem de contribuição.<br>
                        Gráfico comparativo de Receita vs Custo gerado no anexo final do relatório...
                    </div>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; width: 100%;">
                        <h3 style="color: #FFFFFF; font-weight: 900; margin: 0; text-shadow: 0px 4px 15px rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; gap: 10px;">{ICO_LOCK} Relatório Protegido</h3>
                        <p style="color: #FF8A00; font-size: 14px; font-weight: 600; text-shadow: 0px 2px 10px rgba(0,0,0,0.8);">Gráficos Duplos e KPIs Finalizados</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.link_button("Desbloquear Plano Completo — R$ 97,00", url="https://link.mercadopago.com.br/novusai", use_container_width=True)
                
                st.markdown("<div class='btn-secundario'>", unsafe_allow_html=True)
                with open("NOVUS_AI_Estrategia.pdf", "rb") as f:
                    st.download_button("Baixar Relatório (Admin / Teste)", data=f.read(), file_name="NOVUS_AI_Estrategia.pdf", mime="application/pdf")
                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 7. RODAPÉ INTERATIVO
# ==========================================
st.markdown("<br><br><hr style='border-color: #1E1E26; margin-bottom: 30px;'>", unsafe_allow_html=True)

html_rodape = (
    '<div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; margin-bottom: 30px;">'
    f'<div class="badge" style="margin-bottom: 0; min-width: 200px;">{ICO_CHECK}<div><b>Ambiente Seguro</b><br><span style="font-size:10px; color:#64748B;">Criptografia SSL de ponta a ponta</span></div></div>'
    f'<div class="badge" style="margin-bottom: 0; min-width: 200px;">{ICO_SHIELD}<div><b>Proteção LGPD</b><br><span style="font-size:10px; color:#64748B;">Conformidade total com a lei</span></div></div>'
    f'<div class="badge" style="margin-bottom: 0; min-width: 200px;">{ICO_STAR}<div><b>Qualidade Verificada</b><br><span style="font-size:10px; color:#64748B;">Auditoria avançada por IA</span></div></div>'
    '</div>'
)
st.markdown(html_rodape, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #64748B; font-size: 12px;'>© 2026 NOVUS AI. Todos os direitos reservados.</p>", unsafe_allow_html=True)
