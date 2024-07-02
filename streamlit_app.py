import streamlit as st
import time
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

def get_first_llm_response(llm):
    template = "Ask me a question that will help me narrow down my next travel destination"
    prompt_template = PromptTemplate.from_template(template)
    response = llm.predict("Ask me a question that will help me narrow down my next travel destination")
    return response.strip().strip('\"')
    
def get_llm_chain(llm):
    prompt_template = PromptTemplate(    
        template="""= You are a chatbot having a conversation with a human. 
                You will advise the human on choosing a travel destination

        Previous conversation regarding travel preference: {chat_history}
        
        Ask the user a new question to help refine their travel destination:
        """,
        input_variables=["chat_history"]
    )
    
    
    # Notice that we need to align the `memory_key`
    #memory = ConversationBufferMemory(memory_key="chat_history")
    conversation = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
        #memory=memory
    )
    return conversation

def get_llm_chain_from_session() -> LLMChain:
    return st.session_state['llm_chain']

def reset_state():
    if st.session_state['count']:
        del st.session_state['count']
    if st.session_state['llm_chain']:
        del st.session_state['llm_chain']
    if st.session_state['messages']:
        del st.session_state['messages']
    if st.session_state['next_question']:
        del st.session_state['next_question']

# main code
st.title('Travel assistant')

# get open AI key from user
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    st.button('New search', on_click=reset_state)

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
    
# initialize Open AI
import os
os.environ['OPENAI_API_KEY'] = openai_api_key
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature = 0.6)


# Initialize Chat history
messages = []
prompt = ""
count = 0
next_question = ""
llm_chain = None

# initialize and setup session state
if 'llm_chain' not in st.session_state:
    llm_chain = get_llm_chain(llm)
    st.session_state['llm_chain'] = llm_chain
else:
    llm_chain = st.session_state['llm_chain']
    
if 'count' not in st.session_state:
    st.session_state['count'] = 0

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
else:
    messages = st.session_state['messages']

st.text("I am your travel assistant. Let's help you choose your next travel destination")
st.text(count)

# let the user know what we intend to do if they are interacting with this for the first time
if st.session_state['count'] == 0:
    next_question = get_first_llm_response(llm)
    messages.append(next_question)
    st.session_state['messages'] = messages
    st.session_state['next_question'] = next_question.strip()
    st.session_state['count'] = st.session_state['count'] + 1

# check if we need to get more input from the user
if st.session_state['count'] < 3:
    st.text("bbbb: " + st.session_state['next_question'])
    prompt = st.text_input(st.session_state['next_question'])
    if prompt:
        st.session_state['messages'] = st.session_state['messages'].append(prompt)
        st.session_state['count'] = st.session_state['count'] + 1
        #messages = st.session_state['messages']
        #st.text(messages)
        #history = '\n'.join(messages)
        history = st.session_state['messages']
        st.text(history)
        next_question = llm_chain.invoke({"chat_history" : history})["text"]
        st.session_state['next_question'] = next_question.strip()
        st.session_state['messages'] = st.session_state['messages'].append(next_question.strip())
        st.text(st.session_state['next_question'])

if st.session_state['count'] > 100:
    # lets let the user know their travel options
    #response = generate_travel_options(st.session_state['llm_chain'], messages)
    #travel_options = response['travel'].strip().split(",")
    st.write("** Top destinations for you **")
    #st.write(messages)
    #for name in travel_options:
    #    st.write("--", name)
    del st.session_state['count']
    del st.session_state['llm_chain']
    del st.session_state['messages']
    del st.session_state['next_question']
