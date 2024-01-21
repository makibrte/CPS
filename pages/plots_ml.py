import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
def format_filename(filename):
    """
    Formats the given filename to remove special characters that
    might interfere with directory structures.
    """
    # Replace forbidden characters with an underscore
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

    return filename
def main():
    st.header('Select a variable')
    df = pd.read_csv('data/data.csv')
    variables = df.columns[4:21].to_list()
    y_actual = torch.load('actual.pt')
    y_pred = torch.load('pred.pt')
    variable = st.selectbox(
        'Choose a variable',
        tuple(variables)
    )
    st.image(f'plots/lgbm_{format_filename(variable)}.png')
    st.image(f'plots/svr_{format_filename(variable)}.png')
    st.image(f'plots/rf_{format_filename(variable)}.png')

if __name__ == "__main__":
    main()