import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.header('Select a variable')
    df = pd.read_csv('data/data.csv')
    variables = df.columns[4:20].to_list()
    y_actual = torch.load('actual.pt')
    y_pred = torch.load('pred.pt')
    variable = st.selectbox(
        'Choose a variable',
        tuple(variables)
    )
    fig, ax = plt.subplots()
    idx = variables.index(variable)
    ax.boxplot([y_actual[:,idx], y_pred[:,idx]], labels=['Actual', 'Predicted'])
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()