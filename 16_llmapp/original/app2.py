# 必要なモジュールをインポート
import os
from dotenv import load_dotenv, find_dotenv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

#環境変数の読み込み
# find_dotenv() を使うと、上の階層を探して自動的に .env ファイルを見つけてくれます！
load_dotenv(find_dotenv())

# .env から API_KEY を取得して OPENAI_API_KEY にセットする
if "API_KEY" in os.environ:
    os.environ['OPENAI_API_KEY'] = os.environ['API_KEY']
else:
    print("【エラー】.envファイルから API_KEY が見つかりませんでした。")
load_dotenv("../../.env")
os.environ['OPENAI_API_KEY'] = os.environ['API_KEY']
MODEL_NAME ="gpt-4o-mini"

#2.RAG関数の定義
# ※ ./data/text フォルダにテキストファイルを配置しておいてください

try:
    documents = SimpleDirectoryReader('./data/text').load_data()
    rag_index = VectorStoreIndex.from_documents(documents)
    rag_engine = rag_index.as_query_engine()

except Exception as e:
    rag_engine = None

def search_internal_documents(query:str)->str:
    """
    ローカルのドキュメント(./data/text)から社内規定や情報を検索します。
    社内のルールや手続きに関する質問には必ずこれを使用してください。
    """
    if not rag_engine:
        return "ローカルドキュメントが見つかりませんでした。"
    response = rag_engine.query(query)
    return str(response)

#3.Web検索ツールの定義
web_search_tool = TavilySearchResults(max_results = 2)
web_search_tool.name ="web_search"
web_search_tool.description = "インターネット上の最新情報やニュースを検索します。世界の出来事について調べる際に利用します。"

#作成した関数とWeb検索ツールのリストにまとめる
tools = [search_internal_documents, web_search_tool]

#4.キャラクターの設定、グラフの構築
llm = ChatOpenAI(model_name = MODEL_NAME,temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    message:Annotated[list,add_messages]

def build_graph():
    graph_builder = StateGraph(State)

    system_prompt = SystemMessage(
        content="あなたは優秀で丁寧なAIアシスタントです。社内情報や規定に関する質問にはRAGツールを、最新の世の中の情報にはWeb検索ツールを使い分けてユーザーをサポートしてください。言葉使いは丁寧で上品にしてください。"
    )

    def chatbot(state:State):
        messages = state["messages"]
        #先頭にシステムプロンプト（キャラクターの設定）が含まれていなければ追加

        if not any(isinstance(m,SystemMessage)for m in messages):
            messages = [system_prompt] + messages
        return{"messages":[llm_with_tools.invoke(messages)]}

    graph_builder.add_node("chatbot",chatbot)
    graph_builder.add_node("tools",ToolNode(tools))

    graph_builder.add_conditional_edges("chatbot",tools_condition)
    graph_builder.add_edge("tools","chatbot")
    graph_builder.set_entry_point("chatbot")

    return graph_builder.compile(checkpointer=MemorySaver())

graph = build_graph()


#5.チャット実行ループ
print("----- オリジナルチャットボットが起動しました -----")
while True:
    user_input = input("質問：")
    if user_input.strip() == "":
        print("ありがとうございました！")
        break

    config = {"configurable":{"thread_id":"1"}}
    events = graph.stream({"messages":[HumanMessage(content=user_input)]},config,stream_mode="values")

    for event in events:
        if "messages" in event and event["messages"]:
            latest_message = event["messages"][-1]
            if hasattr(latest_message,"content")and latest_message.type == "ai" and latest_message.content:
                print(f"AI:{latest_message.content}\n")
