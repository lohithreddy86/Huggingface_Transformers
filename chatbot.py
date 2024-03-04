import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.title("GEMINI BOT")
#with st.chat_message("user"):
    #st.write("Hello 👋")

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
     

# To display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])    
    

    
#prompt = st.chat_input("please input your question to GEMINI")


if prompt := st.chat_input("Whats up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    #st.write(f"You: \n\n {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})





chat = [
    { "role": "user", "content": prompt },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=2500)


 
if prompt is not None:
    response = f"Echo: {tokenizer.decode(outputs[0])}"

    
with st.chat_message("assistant"):
    st.markdown(response)
    
#add response to session state messages
st.session_state.messages.append({"role": "assistant", "content": response})            
    
# streamlit run /home/dlvaayuai/storage/NLP_LP_V2/Huggingface_Transformers/chatbot.py   