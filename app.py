import pandas as pd
import streamlit as st
import utils

# Function to display the dictionary values except 'None'
def display_dict_values(data):
    for key, value in data.items():
        if value != "None":
            st.markdown(f"**{key}:** {value}")

# Function to display items as a dataframe
def display_items_as_dataframe(items):
    if items:  # Check if the items list is not empty
        df = pd.DataFrame(items)
        st.dataframe(df)

# Display a title
st.title("Invoice Buddy")

# Short description of the app
st.markdown("""
### Extract information from invoices
""")

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload invoice image (png, jpg, jpeg)", type=['png', 'jpg'])

if uploaded_file is not None:
    utils.empty_directory('data')
    utils.save_uploaded_file('data', uploaded_file)
    response = utils.pass_to_openai_vision_api_llama_index(uploaded_file.name)
    st.markdown('Data extracted from invoice is ')
    # Display dictionary values
    display_dict_values({key: value for key, value in response.items() if key != "Items"})

    # Display items in a dataframe
    if "Items" in response and response["Items"]:
        st.markdown("**Items:**")
        display_items_as_dataframe(response["Items"])
    utils.empty_directory('data')