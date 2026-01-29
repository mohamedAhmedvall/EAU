#!/bin/bash
# Lancer l'application Streamlit Optiplan

cd "$(dirname "$0")"
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
