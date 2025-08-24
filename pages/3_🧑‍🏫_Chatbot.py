import streamlit as st
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain_core.prompts.prompt import PromptTemplate
import os
import streamlit as st
import base64
from huggingface_hub import InferenceClient

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

# Custom Llama LLM class for Hugging Face API
class HuggingFaceLlamaLLM:
    def __init__(self, api_key, model_name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=3600, temperature=0.5):
        self.client = InferenceClient(
            provider="fireworks-ai",
            api_key=api_key,
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def __call__(self, prompt):
        return self.generate(prompt)
    
    def generate(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling Hugging Face API: {str(e)}")
            return "Sorry, I encountered an error processing your request."

# Session state initialization
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
    """Clears session state and starts a new chat."""
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    # Reset memory if exists
    if hasattr(st.session_state, 'conversation_history'):
        st.session_state.conversation_history = []

K = 7
# Replace with your Hugging Face API key
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in your environment

st.subheader("Chat with AI Advisor to get a brief about stock related terms.")

# Initialize Llama model
llm = HuggingFaceLlamaLLM(api_key=HF_TOKEN, temperature=0.5, max_tokens=3600)

# Custom conversation handler
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def generate_response(user_input, history):
    # Build conversation context
    system_prompt = """As a finance advisor, your task is to familiarize clients with stock and investment-related terms in a concise and engaging manner. Through this role play, aim to simplify complex concepts and encourage active participation to ensure comprehension and confidence in navigating financial discussions. Your expertise lies in assessing risk levels and tailoring investment strategies accordingly."""
    
    # Build conversation context
    context = system_prompt + "\n\nConversation History:\n"
    
    # Add conversation history
    for entry in history[-K:]:  # Keep last K entries
        context += f"{entry}\n"
    
    context += f"\nHuman: {user_input}\nAssistant:"
    
    response = llm.generate(context)
    return response

# New Chat button
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

st.write(st.session_state["cnt"])

if st.session_state["cnt"] == 0:
    output = generate_response("Hi. Introduce Yourself.", st.session_state.conversation_history)
    st.session_state.past.append("")  
    st.session_state.generated.append(output)
    st.session_state.conversation_history.append(f"Assistant: {output}")
    st.session_state["cnt"] += 1

if st.session_state["cnt"] >= 1:
    user_input = get_text()
    submit_button = st.button("Generate")
    if submit_button:
        if user_input:
            st.session_state.conversation_history.append(f"Human: {user_input}")
            output = generate_response(user_input, st.session_state.conversation_history)
            st.session_state.past.append(user_input)  
            st.session_state.generated.append(output)
            st.session_state.conversation_history.append(f"Assistant: {output}")

# Display conversation
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i], icon="😊")
        st.success(st.session_state["generated"][i], icon="🤖")

# Display stored sessions
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
