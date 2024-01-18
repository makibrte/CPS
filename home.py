import streamlit as st
import matplotlib.pyplot as plt
import torch

def main():
    st.title('Soil extracts data analysis')
    st.header('Data preparation')
    st.text('The model was fitted on soil extract data.')
    st.text('The model tries to predict chemical properties of soil using taxa data')
    st.text('For data preparation PCA was used in order to reduce dimensionality due to n<p issue')
    st.text('After calculating PCA 16 components are used to describe the taxa data.')
    st.header('Model structure and training')
    st.text('PyTorch was used to create a NN model. Input size is 16 and output size is 17.')
    st.subheader('Model layers')
    st.text('Linear(16,100) -> ReLU -> LayerNorm(100) -> Linear(100,50)-> ReLU -> Linear(50,17)')
    st.subheader('Training')
    st.text('Model was trained in batches of size 8 with Adam optimizer with lr of 0.01')
        

if __name__ == "__main__":
    main()