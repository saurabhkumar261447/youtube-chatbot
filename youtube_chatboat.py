import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# CONFIG
# ----------------------------
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# HF_TOKEN = "hf_akPUbrODTZGELGiFiQTUiHeNArmGeRJc"
EMBEDDING_MODEL = "skzentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# ----------------------------
# LOAD YOUTUBE TRANSCRIPT
# ----------------------------
# def load_transcript(video_id: str, language: str = "auto"):
#     if language == "auto":
#         lang_codes = ["hi"]
#         translation = "en"
#     elif language == "en":
#         lang_codes = ["hi"]
#         translation = "en"
#     else:
#         lang_codes = [language]
#         translation = None
# 
#     loader = YoutubeLoader.from_youtube_url(
#         f"https://www.youtube.com/watch?v={video_id}",
#         add_video_info=False,
#         language=lang_codes,
#         translation=translation
#     )
#     docs = loader.load()
#     return docs


def load_transcript(video_id: str, language: str = "auto"):
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Language resolution
    if language == "auto":
        languages = ["en", "hi"]
    else:
        languages = [language]

    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        language=languages
    )

    docs = loader.load()
    return docs

# ----------------------------
# BUILD VECTOR STORE
# ----------------------------
def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        split_docs,
        embedding=embeddings
    )

    return vectorstore

# ----------------------------
# BUILD RAG CHAIN
# ----------------------------
def build_qa_chain(vectorstore):
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            task="conversational",
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=256
        )
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the following question based only on the provided context:\n\n{context}"),
        ("human", "{question}")
    ])

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

# ----------------------------
# TERMINAL CHAT
# ----------------------------
def chat_loop(qa_chain):
    print("\n🤖 YouTube RAG Chatbot Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = qa_chain.invoke(query)
        print(f"\nBot: {result}\n")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    video_id = input("Enter YouTube Video ID: ").strip()
    language = input("Transcript language (en / hi / auto): ").strip() or "auto"


    print("\n📥 Loading transcript...")
    documents = load_transcript(video_id, language)
    print("📊 Building vector store...")
    vectorstore = build_vector_store(documents)

    print("🧠 Initializing RAG pipeline...")
    qa_chain = build_qa_chain(vectorstore)

    chat_loop(qa_chain)
