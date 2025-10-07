from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

# モデルのロード (量子化モデルを指定、例: ggufファイル)
llm = LlamaCpp(
    model_path="C:/Users/omede/.lmstudio/models/lmstudio-community/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf",  # ここを自分のモデルに変更
    n_ctx=2048,     # コンテキストサイズ
    n_threads=6,    # CPUスレッド数
    n_batch=512,    # 推論時のバッチサイズ
    verbose=True,   # デバッグログ出力
)

# プロンプトテンプレート
prompt = PromptTemplate.from_template(
    "次の質問に日本語で簡潔に答えてください:\n\n{question}"
)

# 実行
question = "自己概念と自己効力感の違いは何ですか？"
response = llm.invoke(prompt.format(question=question))
print("回答:", response)
