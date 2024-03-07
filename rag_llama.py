
from torch import cuda, bfloat16
import transformers
import torch
import locale
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

#Enter your Hugging Face Token Access here
hf_auth = 'Your_Hugging_Face_Token_Access'

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

model.eval()
# print(f"Model loaded on {device}")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
# print(stop_token_ids)
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
# print(stop_token_ids)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False
stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    stopping_criteria=stopping_criteria,
    temperature=0.1,
    max_new_tokens=512,
    repetition_penalty=1.1
)
#Generate Text from Query
res = generate_text("Explain me the difference between Data Lakehouse and Data Warehouse.")
print(res[0]["generated_text"])

llm = HuggingFacePipeline(pipeline=generate_text)
#Generate LLM response
print(llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse."))

locale.getpreferredencoding = lambda: "UTF-8"

loader = PyPDFLoader("/content/document.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

vectorstore = FAISS.from_documents(all_splits, embeddings)
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
chat_history = []

#Chat with Loaded Document
query = "What is the age restriction for a licensee to handle cannabis products?"
result = chain({"question": query, "chat_history": chat_history})
print(result['answer'])

#For History
chat_history = [(query, result["answer"])]
query = "What is the age restriction for a licensee to handle cannabis products?"
result = chain({"question": query, "chat_history": chat_history})
print(result['answer'])


#Check the source of document that is used to generate the above answer
print(result['source_documents']) 