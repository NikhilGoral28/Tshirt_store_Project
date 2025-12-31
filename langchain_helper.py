from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma 
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
import os
from dotenv import load_dotenv
# NEW (correct)
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



from few_shots import few_shots
#api key
load_dotenv()
api_key = os.environ["api_key"]

def get_few_shot_db_chain():
#creating a llm
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.7
    )

#db object
    db_user = "root"
    db_password = ""
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db =SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info = 3)


    #embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    )

    #vectorize
    to_vectorize = ["" .join(example.values()) for example in few_shots]

    #creating vector db
    vectorstore = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas=few_shots)

    #creating selector 
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2
    )

    #template
    example_prompt = PromptTemplate(
        input_variables= ["Question", "SQLQuuery", "SQLResult", "Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",

    )

    #creating few shot prompt
    few_shot_prompt =FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = _mysql_prompt,
        suffix = PROMPT_SUFFIX,
        input_variables = ["input","table_info","top_k"],
    )
    
    chain = SQLDatabaseChain.from_llm(llm, db, verbose= True, prompt=few_shot_prompt)

    return chain

"""

if __name__ == "__main__":
    chain = get_few_shot_db_chain()
    print(chain.run("How many t shirts are left in total stock?P"))

"""