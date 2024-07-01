import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

def get_llm_chain():
    template = """You are a chatbot having a conversation with a human. 
                You will advise the human on choosing a travel destination
                If there is no prior chat history, then ask a random question to help narrow
                down the travel destination.

    Previous conversation regarding travel preference:
    {chat_history}

    Human response based on conversation: {response}
    
    New question to human based on conversation:
    Question:"""

    prompt_template = PromptTemplate.from_template(template)
    
    # Notice that we need to align the `memory_key`
    memory = ConversationBufferMemory(memory_key="chat_history")
    conversation = LLMChain(
        llm=get_llm(),
        prompt=prompt_template,
        verbose=True,
        memory=memory
    )
    return conversation

def get_llm_chain_from_session() -> LLMChain:
    return st.session_state['llm_chain']
    

def generate_baby_names(gender: str,nationality:str) -> list[str]:
    """
    Generate a list of 5 baby names

    Parameters:
    gender (str): gender of baby
    nationailty (str) : nationailty of baby

    Returns:
    list: list of baby names
    """

    prompt_template_name = PromptTemplate(
        input_variables=['gender', 'nationality'],
        template="""I want to find a name for a {nationality} {gender} baby. 
                    Suggest top 5 popular names for the baby.
                    Return it as a comma separated list """
                )

    name_chain = LLMChain(llm=llm,
                          prompt=prompt_template_name,
                          output_key='baby_names')

    chain = SequentialChain(
        chains=[name_chain],
        input_variables=['gender', 'nationality'],
        output_variables=['baby_names']
    )

    response = chain({'gender': gender,
                      'nationality': nationality})
    return response


# main code
st.title('Travel assistant')

# get open AI key from user
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    
# initialize Open AI
import os
os.environ['OPENAI_API_KEY'] = openai_api_key
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature = 0.6)


# Initialize Chat history
messages = []
prompt = None
count = 0

# initialize and setup session state
if 'llm_chain' not in st.session_state:
    st.session_state['llm_chain'] = get_llm_chain()
if 'count' not in st.session_state:
    st.session_state['count'] = 0
else:
    count = st.session_state['count']
    
if 'messages' not in st.session_state:
    st.session_state['messages'] = messages
else:
    messages = st.session_state['messages']
if 'prompt' in st.session_state:
    prompt = st.session_state['prompt']

# let the user know what we intend to do if they are interacting with this for the first time
if st.session_state['count'] == 0:
    st.text("I am your travel assistant. Let's help you choose your next travel destination")

# check if we need to get more input from the user
if count <= 5:
    next_question = llm_chain({"chat_history" : messages, "response": prompt})["text"]
    st.session_state['prompt'] = st.chat_input(next_question)
    messages.append(next_question)
    messages.append(st.session_state['prompt'])
    st.session_state['messages'] = messages
    st.session_state['count'] = count + 1
else:
    # lets let the user know their travel options
    #response = generate_travel_options(st.session_state['llm_chain'], messages)
    #travel_options = response['travel'].strip().split(",")
    st.write("** Top destinations for you **")
    st.write(messages)
    #for name in travel_options:
    #    st.write("--", name)
    del st.session_state['count']
    del st.session_state['llm_chain']
    del st.session_state['messages']
    del st.session_state['prompt']
