import azure.functions as func
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
import os
import time


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:

    openAI_API_key = os.environ.get("OPENAI_API_KEY")
    pinecone_API_key = os.environ.get("PINECONE_API_KEY")

    query = req.params.get("query")

    if query == "" or query == None:
        query = "Tell me about yourself"

    embeddings = OpenAIEmbeddings(api_key=openAI_API_key)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="personal-website", embedding=embeddings
    )
    llm = ChatOpenAI(api_key=openAI_API_key, temperature=1, model="gpt-4-turbo")

    rag_prompt = """
    Here is the following information about someone you'll be impersonating. ALWAYS answer as if you are him. THIS IS REALLY IMPORTANT. Remember to be polite and super casual. If the information is not listed out below DO NOT make up any information. If you are unsure, deflect and encourage the person to ask more about the person in this context!

    PRETEND THAT IT IS DECEMBER 2024.

    {context}

    Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(rag_prompt)

    retriever = vectorstore.as_retriever()

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    response: AIMessage = rag_chain.invoke(query)

    # return response.content

    logging.info("Python HTTP trigger function processed a request.")

    return func.HttpResponse(
        response.content,
        status_code=200,
    )
