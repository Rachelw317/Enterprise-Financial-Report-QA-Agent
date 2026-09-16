# Enterprise Financial Report QA Agent

基于企业财务报告 PDF 和向量检索的问答 Agent。系统会分析用户问题，必要时拆分复杂问题，逐个检索证据，并对检索结果和最终答案进行验证。

## 准备环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

设置 DeepSeek API Key：

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

也可以将配置写入项目根目录的 `.env` 文件。

## 建立向量库

将财务报告 PDF 放入 `data/raw/`，然后运行：

```bash
python scripts/ingest.py
```

## 运行问答

交互模式：

```bash
python main.py
```

单次提问：

```bash
python main.py "比较 2022 年和 2023 年的营业收入"
```

常用参数：

```bash
python main.py \
	--vectorstore-directory data/vectorstore \
	--collection-name financial_reports \
	--model deepseek-chat \
	--max-retries 2 \
	"2023 年的净利润是多少？"
```
