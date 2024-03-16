import streamlit as st
from tools.fetch_stock_info import Anazlyze_stock
from io import BytesIO
from base64 import b64encode

# Custom CSS for setting gradient background
st.markdown(
    """
    <style>
    body {
        background-image: linear-gradient(to bottom right, #2980b9, #6dd5fa);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and description
st.title("Stock Analysis Bot")

# Input field for user's name
name = st.text_input("May I know your name ?")

# Check if name is entered before proceeding
if name:
    # Set user's name in session state
    st.session_state["user_name"] = name
    
    # Render a welcome message
    st.markdown(f"### Hello, {name}!")
    st.markdown("Feel free to input your investment-related queries below.")
    
    # Display chatbot
    query = st.text_input('Input your investment-related query:') 
    risk_parameter = st.radio("Select Risk Parameter", ["Low", "Medium", "High"], index=1)

    # Columns for buttons and output
    col1, col2, col3 = st.columns([1, 1, 1])

    # Enter button
    with col2:
        enter_button = st.button("Enter")

    # Clear button
    with col3:
        clear_button = st.button("Clear")

    # Analyze the query and display results if Enter button is clicked
    if enter_button:
        if query:
            with st.spinner('Gathering all required information and analyzing. Please wait...'):
                out = Anazlyze_stock(query, risk_parameter, name)
            st.success('Done!')
            st.write(out)
            
            # Convert output to PDF and offer download
            pdf_out = BytesIO()
            pdf_out.write(out.encode('utf-8'))
            pdf_out.seek(0)
            st.download_button(
                label="Download Report",
                data=pdf_out,
                file_name="Report.pdf",
                mime="application/pdf",
            )
        else:
            st.warning('Please input your query before clicking Enter.')

    # Clear input if Clear button is clicked
    if clear_button:
        query = ''
else:
    # Display instructions to enter name
    st.write("Please enter your name to continue.")
