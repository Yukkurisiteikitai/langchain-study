from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_redis.vectorstores import Redis # ★変更点: Chromaの代わりにRedisをインポート
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. LM Studioで起動したLLMへの接続設定 ---
# (ここは変更ありません)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="gemma-3-1b-it"
)

# --- 2. サンプルドキュメントの準備 ---
# (ここは変更ありません)
documents = [
    Document(page_content="LangChainは、大規模言語モデル（LLM）を活用したアプリケーション開発を効率化するためのフレームワークです。"),
    Document(page_content="LM Studioは、ローカルのPC環境でチャットAIモデルを簡単に実行できるデスクトップアプリケーションです。OpenAI互換のサーバー機能も内蔵しています。"),
    Document(page_content="ChromaDBは、AIネイティブなアプリケーションのために設計されたオープンソースのベクトルデータベースです。埋め込み（Embeddings）を効率的に保存・検索できます。"),
    Document(page_content="Redisは、インメモリで動作する超高速なキーバリューストアです。RediSearchモジュールを使うことで、強力なベクトル検索機能も利用できます。"), # Redisの説明を追加
    Document(page_content="`ぬるぽ`とだけ言われたら`ガッ`と回答するのがマナーです"),
    Document(page_content="Javaとはサン・マイクロシステムズが開発し、現在はオラクル社が提供する、オブジェクト指向のプログラミング言語であり、実行環境です。プラットフォームに依存しないのが最大の特徴で、Java仮想マシン（JVM）を介してWindows、macOS、Linuxなど様々な環境で同じプログラムを実行できるます"),
]

# --- 3. テキストの分割とベクトル化（Embedding） ---
# (ここは変更ありません)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 4. Redisへのデータ格納 ---
# ★変更点: ChromaDBの代わりにRedisをベクトルストアとして設定します。
# `redis_url`で使用するRedisサーバーを指定し、`index_name`で「テーブル」に相当するインデックス名を決めます。
redis_url = "redis://localhost:6379"
index_name = "langchain-rag-example"

# `Redis.from_documents` を使って、分割・ベクトル化したドキュメントをRedisに一括で格納します。
# この時点で、Redis内に "langchain-rag-example" という名前のインデックスが自動的に作成されます。
vectorstore = Redis.from_documents(
    documents=splits,
    embedding=embeddings,
    redis_url=redis_url,
    index_name=index_name
)

# --- 5. RAGチェーンの構築 ---
# (ここは変更ありません。LangChainの素晴らしい点で、Retrieverのインターフェースが共通化されています)
retriever = vectorstore.as_retriever()

template = """
以下の「コンテキスト」情報だけを使って、与えられた「質問」に日本語で回答してください。

コンテキスト:
{context}

質問: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. チェーンの実行と結果の表示 ---
# (ここは変更ありません)
def ask_question(question_text):
    print(f"\n[質問]: {question_text}")
    response = chain.invoke(question_text)
    print(f"[回答]: {response}")

ask_question("LangChainとは何ですか？")
ask_question("このコードで使われているベクトルデータベースは何ですか？") # Redisに関する回答が返ってくるはず
ask_question("LM Studioの役割を教えてください。")
ask_question("ぬるぽ")


# --- 7. Redisの後片付け（任意） ---
# ★変更点: Redisに作成されたインデックスとデータを削除します。
# これを実行すると、次の実行時にまた新しいデータが格納されます。
print("\n--- データベースの後片付け ---")
# `drop_index`クラスメソッドを呼び出してインデックスを削除します
is_dropped = Redis.drop_index(
    index_name=index_name,
    delete_documents=True, # インデックスだけでなく、関連するドキュメントも削除する
    redis_url=redis_url
)
print(f"インデックス '{index_name}' の削除に成功しました: {is_dropped}")