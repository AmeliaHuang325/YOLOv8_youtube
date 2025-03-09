import streamlit as st
import torch

st.write(f"Python Path: {torch.__file__}")
st.write(f"PyTorch Version: {torch.__version__}")
st.write(f"CUDA Available: {torch.cuda.is_available()}")
