import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate

st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
)
import streamlit as st
import base64  # Import the base64 module

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# add_bg_from_local('bala.jpeg')  # Replace with your image file path


if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


if ("username" not in st. session_state) and ("mood" not in st.session_state) and ("reason" not in st.session_state):
    st.session_state["username"] = ""
    st.session_state["mood"] = ""
    st.session_state["reason"] = ""
def get_text():
    
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Enter 'Hi' to start conversation with Bot. ", 
                            label_visibility='hidden')

    return input_text
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()

K = 30
MODEL = "gpt-3.5-turbo-16k"
st.title("StockSageBot")
st.subheader("Stock Market Recommender")
name = st.session_state["username"]
mood = st.session_state["mood"]
reason = st.session_state["reason"]
API_O = " "


CUSTOM_CONVERSATION_TEMPLATE = (
    f"You are Leon, a seasoned financial advisor for investors, with years of experience in navigating the complexities of the stock market. Start the conversation with user by introducing yourself and asking their name. As the financial advisor, your task is to engage in a detailed discussion with the client, understanding their financial objectives, risk tolerance, specific preferences they may have regarding their investments and guide investors in selecting the most suitable stocks for their unique financial goals and risk profiles.You must draw upon your expertise to provide unbiased recommendations tailored to their unique circumstances, offering insights into market conditions and potential opportunities while managing expectations and mitigating risks. The goal is to establish a trusting relationship built on transparency and sound financial advice, guiding the client towards achieving their financial aspirations.\n"
    "Context:\n"
    "{entities}\n"
    "Current conversation:\n"
    "{history}\n"
    "Last line:\n"
    "Human: {input}\n"
    "You:"
)


if API_O:
    llm = ChatOpenAI(temperature=0,
                openai_api_key=API_O, 
                model_name=MODEL, 
                verbose=False,max_tokens=2000) 



    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
            print("Memory: " , st.session_state.entity_memory)

    Conversation = ConversationChain(
            llm=llm, 
            prompt=PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=CUSTOM_CONVERSATION_TEMPLATE,
)
,
            memory=st.session_state.entity_memory

        )  
    print(st.session_state.entity_memory)

###### New Chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# if (st.session_state["username"]!= "") and (st.session_state["mood"]!= "") and (st.session_state["reason"] != ""):
user_input = get_text()
submit_button = st.button("Generate")
if submit_button:
    if user_input:
        output = Conversation.run(input=user_input)  
        st.session_state.past.append(user_input)  
        st.session_state.generated.append(output)  
#### Download conversation
 
download_str = []

with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="😊")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    download_str = '\n'.join(download_str)
    if download_str:
        st.sidebar.download_button('Download',download_str)

for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
