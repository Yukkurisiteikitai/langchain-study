import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは料理専門家です"),
        ("user", "{input}"),
    ]
)

output_parser = StrOutputParser()

# LCELでパターンごとに結果を確認する
chain1 = prompt
chain2 = prompt | llm
chain3 = prompt | llm | output_parser

response1 = chain1.invoke({"input": "美味しい和食といえば？１つあげて。"})
print(response1)
print("------------")
response2 = chain2.invoke({"input": "美味しい和食といえば？１つあげて。"})
print(response2)
print("------------")
response3 = chain3.invoke({"input": "美味しい和食といえば？１つあげて。"})
print(response3)
print("------------")