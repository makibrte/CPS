import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.header('Select a variable')
    df = pd.read_csv('data/data.csv')
    variables = df.columns[4:21].to_list()
    y_actual = torch.load('actual_rmsle.pt')
    y_pred = torch.load('pred_rmsle.pt')
    variable = st.selectbox(
        'Choose a variable',
        tuple(variables)
    )
    fig, ax = plt.subplots()
    idx = variables.index(variable)
    loss = torch.nn.MSELoss()
    ax.boxplot([y_actual[:,idx], y_pred[:,idx]], labels=['Actual', 'Predicted'])
    st.write('RMSE Loss for this plot is : ', np.sqrt(loss(y_actual[:,idx], y_pred[:, idx]).item()))
    st.pyplot(fig)

if __name__ == "__main__":
    main()