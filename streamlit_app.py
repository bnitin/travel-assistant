import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

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
        
    conversation = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )
    return conversation

def get_llm_chain_from_session() -> LLMChain:
    return st.session_state.llm_chain

def get_next_question(llm, count, chat_history, max_questions):
    if count == 0:
        return get_first_llm_response(llm)    
    elif count <= max_questions:
        llm_chain = get_llm_chain_from_session()
        history = '\n'.join(chat_history)        
        return llm_chain.invoke({"chat_history" : history})["text"]
    else:
        return None
    
def reset_state():
    if 'count' in st.session_state:
        del st.session_state.count
    if 'llm_chain' in st.session_state:
        del st.session_state.llm_chain
    if 'chat_history' in st.session_state:
        del st.session_state.chat_history

def update_prompt():
    prompt = st.session_state.text_key
    chat_history = st.session_state.chat_history
    chat_history.append(prompt)
    st.session_state.chat_history = chat_history

def initialize_llm():
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
    
    return llm

def generate_travel_options(llm, user_pref, question_count):
    prompt_template = PromptTemplate(
        input_variables=['question_count', 'user_pref'],
        template = """I would like to go on vacation. Use these {} questions and answers to 
                        determine 3 places for me to travel to. 
                        Write a list of activities that we can do at each of these spots using these 
                        questions and answers Also write a brief paragraph summarizing the places.

                        here are the questions: {}
                    """.format(question_count, user_pref)
    )

    name_chain = LLMChain(llm=llm,
                          prompt=prompt_template,
                          output_key='travel')

    chain = SequentialChain(
        chains=[name_chain],
        input_variables=['question_count', 'user_pref'],
        output_variables=['travel']
    )

    response = chain({'question_count' : question_count, 'user_pref': user_pref})
    return response['travel']


############################################

# main code
st.title('Travel assistant')

llm = initialize_llm()

# Initialize Chat history
chat_history = []
prompt = ""
count = 0
next_question = None
llm_chain = None
max_questions = 5

# initialize and setup session state
if 'llm_chain' not in st.session_state:
    llm_chain = get_llm_chain(llm)
    st.session_state.llm_chain = llm_chain
else:
    llm_chain = st.session_state.llm_chain
    
if 'count' not in st.session_state:
    st.session_state.count = 0

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    chat_history = st.session_state.chat_history

st.text("I am your travel assistant. Let's help you choose your next travel destination")

# let the user know what we intend to do if they are interacting with this for the first time
next_question = get_next_question(llm, st.session_state.count, chat_history, max_questions)
    
# check if we need to get more input from the user
if next_question:
    chat_history.append(next_question)
    st.session_state.chat_history = chat_history
    st.session_state.count += 1
    
    # get input from user
    prompt = st.text_input(label=next_question, on_change=update_prompt, key='text_key')

else:
    # lets let the user know their travel options
    travel_options = generate_travel_options(llm, chat_history, max_questions)
    st.write("** Top 3 destinations for you **")
    st.write(travel_options)
    
    # cleanup state
    reset_state()
