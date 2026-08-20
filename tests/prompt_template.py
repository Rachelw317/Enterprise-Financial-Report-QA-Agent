from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system",
     """你是一个专业的知识库问答助手。
请根据提供的资料回答用户的问题。
你的回答必须基于资料中的信息，不要编造资料中没有的信息。
如果资料中没有足够的信息回答问题，请明确告诉用户“根据现有资料无法确定”。
    资料：{context}"""
    ),
    ("human",
    "{question}"
    )
])