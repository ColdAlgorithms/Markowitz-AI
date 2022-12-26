import streamlit as st

st.set_page_config(
    page_title="Financial  App",
    page_icon="💵",
    initial_sidebar_state = "expanded"
    )

def main():
    st.title("Main Page")
    st.sidebar.success("Select a page above.")

if __name__ == "__main__":
    main()