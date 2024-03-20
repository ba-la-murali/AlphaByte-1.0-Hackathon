import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
 
import os
import streamlit as st
import base64  
 
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

# add_bg_from_local('img.gif')  # Replace with your image file path


if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "cnt" not in st.session_state:
    st.session_state["cnt"] = 0

 
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

K = 7
MODEL = "gpt-3.5-turbo"
    
st.subheader("Chat with AI Advisor to get a brief about stock related terms.")
 

 
os.environ["OPENAI_API_KEY"]= os.getenv("OPEN_AI_KEY")


CUSTOM_CONVERSATION_TEMPLATE = (
    f"As a finance advisor, your task is to familiarize clients with stock and investment-related terms in a concise and engaging manner. Through this role play, aim to simplify complex concepts and encourage active participation to ensure comprehension and confidence in navigating financial discussions.Your expertise lies in assessing risk levels and tailoring investment strategies accordingly.\n"
    "Context:\n"
    "{entities}\n"
    "Current conversation:\n"
    "{history}\n"
    "Last line:\n"
    "Human: {input}\n"
    "You:"
)


 
llm = ChatOpenAI(temperature=0.5,
                
            model_name=MODEL, 
            verbose=False,max_tokens=3600) 



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
download_str = []
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

 
st.write(st.session_state["cnt"])
if st.session_state["cnt"] == 0:
        output = Conversation.run(input=f"Hi. Introduce Yourself.")  
        st.session_state.past.append("")  
        st.session_state.generated.append(output)
        st.session_state["cnt"] += 1
if st.session_state["cnt"] >= 1:
    user_input = get_text()
    submit_button = st.button("Generate")
    if submit_button:
        if user_input:
            output = Conversation.run(input=user_input) 
            st.session_state.past.append(user_input)  
            st.session_state.generated.append(output)
             

else:
    st.write("Fill your Name and tell about your current mood.")
download_str = []

with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="😊")
        st.success(st.session_state["generated"][i], icon="🤖")
        # download_str.append(st.session_state["past"][i])
        # download_str.append(st.session_state["generated"][i])

 
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
