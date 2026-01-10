from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# ----------------------------
# Setup environment
# ----------------------------
load_dotenv()
OPENAI_TOKEN = os.getenv("GITHUB_TOKEN")

os.environ["OPENAI_API_KEY"] = OPENAI_TOKEN
os.environ["OPENAI_API_BASE"] = "https://models.github.ai/inference"

# ----------------------------
# Initialize models
# ----------------------------
llm = ChatOpenAI(model="openai/gpt-4.1", temperature=1.0)
translator_llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# ----------------------------
# Translate helper
# ----------------------------
def translate_once(docs):
    text_block = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Translate the following text into English:\n\n{text_block}"
    translation = translator_llm.invoke(prompt).content
    return translation

# ----------------------------
# Prompt
# ----------------------------
template = """You are a helpful assistant.
Answer ONLY from the provided transcript context.
Always respond in English, even if the transcript was originally in another language.
If context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# ----------------------------
# Main function used by GUI
# ----------------------------
def run_qa(video_id: str, question: str):
    # Fetch transcript
    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id, languages=["en", "hi", "fr", "ur"]
        )
        raw_transcript = " ".join(chunk.text for chunk in transcript_list)

    except TranscriptsDisabled:
        return "❌ No captions available for this video."

    except Exception as e:
        return f"❌ Error fetching transcript: {e}"

    if not raw_transcript:
        return "❌ Empty transcript."

    # Split transcript
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents([raw_transcript])

    # Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Build FAISS in batches
    batch_size = 10
    vector_store = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_store = FAISS.from_documents(batch, embedding=embeddings)
        if vector_store is None:
            vector_store = batch_store
        else:
            vector_store.merge_from(batch_store)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    # Translate retrieved chunks
    def format_fun(retrieved_docs):
        return translate_once(retrieved_docs)

    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_fun),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser

    # Run QA
    result = main_chain.invoke(question)
    return result
