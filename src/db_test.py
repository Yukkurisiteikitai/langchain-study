import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. LM Studioで起動したLLMへの接続設定 ---
# LM Studioのローカルサーバーに接続します。
# model名はLM StudioのUIでロードしたモデルに合わせて変更することもできますが、
# "local-model"のままでも通常は動作します。api_keyは不要です。
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="gemma-3-1b-it" 
)

# --- 2. サンプルドキュメントの準備 ---
# ChromaDBに格納するためのテキストデータを用意します。
documents = [
    Document(page_content="LangChainは、大規模言語モデル（LLM）を活用したアプリケーション開発を効率化するためのフレームワークです。"),
    Document(page_content="LM Studioは、ローカルのPC環境でチャットAIモデルを簡単に実行できるデスクトップアプリケーションです。OpenAI互換のサーバー機能も内蔵しています。"),
    Document(page_content="ChromaDBは、AIネイティブなアプリケーションのために設計されたオープンソースのベクトルデータベースです。埋め込み（Embeddings）を効率的に保存・検索できます。"),
    Document(page_content="このサンプルコードでは、これら3つのツールを連携させ、ドキュメントの内容に基づいた質疑応答システム（RAG）を構築します。"),
    Document(page_content="`ぬるぽ`とだけ言われたら`ガッ`と回答するのがマナーです"),
    Document(page_content="Javaとはサン・マイクロシステムズが開発し、現在はオラクル社が提供する、オブジェクト指向のプログラミング言語であり、実行環境です。プラットフォームに依存しないのが最大の特徴で、Java仮想マシン（JVM）を介してWindows、macOS、Linuxなど様々な環境で同じプログラムを実行できるます"),
    
]

# --- 3. テキストの分割とベクトル化（Embedding） ---
# ドキュメントを適切なサイズに分割します。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# テキストをベクトルに変換するためのEmbeddingモデルを準備します。
# ここではローカルで動作するHuggingFaceのモデルを使用します。
# 初回実行時にはモデルのダウンロードが行われます。
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 4. ChromaDBへのデータ格納 ---
# 分割・ベクトル化したドキュメントをChromaDBに格納します。
# persist_directoryを指定すると、ベクトルデータがディスクに保存され、再利用できます。
# 指定しない場合はインメモリで動作します。
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# --- 5. RAGチェーンの構築 ---
# ChromaDBを情報検索のためのRetrieverとして設定します。
retriever = vectorstore.as_retriever()

# LLMに渡すプロンプトのテンプレートを定義します。
template = """
以下の「コンテキスト」情報だけを使って、与えられた「質問」に日本語で回答してください。

コンテキスト:
{context}

質問: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LangChain Expression Language (LCEL) を使ってチェーンを構築します。
# 処理の流れ:
# 1. 質問を受け取り、Retrieverで関連情報を検索する (context)
# 2. 検索結果と元の質問をプロンプトに埋め込む
# 3. プロンプトをLLMに渡して回答を生成する
# 4. LLMの出力を文字列として整形する
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. チェーンの実行と結果の表示 ---
def ask_question(question_text):
    print(f"\n[質問]: {question_text}")
    response = chain.invoke(question_text)
    print(f"[回答]: {response}")

# 質問を投げてみましょう
ask_question("LangChainとは何ですか？")
ask_question("このコードで使われているベクトルデータベースは何ですか？")
ask_question("LM Studioの役割を教えてください。")
ask_question("ぬるぽ")


# --- 7. ChromaDBの後片付け（任意） ---
# インメモリのデータベースをクリーンアップする場合は以下を実行
# vectorstore.delete_collection()