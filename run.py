from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from openai import OpenAI
from FlagEmbedding import FlagAutoModel

from lib import *

load_dotenv()


def llm_run_reasoning(prompt: str, model: str = 'qwq-32b') -> str:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        # enable_thinking 参数开启思考过程，QwQ 与 DeepSeek-R1 模型总会进行思考，不支持该参数
        extra_body={"enable_thinking": True},
        stream=True,
    )

    reasoning_content = ""  # 完整思考过程
    answer_content = ""  # 完整回复
    is_answering = False  # 是否进入回复阶段
    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        # 只收集思考内容
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            if not is_answering:
                print(delta.reasoning_content, end="", flush=True)
            reasoning_content += delta.reasoning_content

        # 收到content，开始进行回复
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
                print("\n" + "=" * 40 + "\n")
            print(delta.content, end="", flush=True)
            answer_content += delta.content

    return answer_content


def llm_run(prompt: str, model: str = 'qwen-plus') -> str:
    if model in ['qwq-32b']: #deepseek-r1
        return llm_run_reasoning(prompt)
    else:
        llm = ChatOpenAI(model=model)
        result = llm.invoke(prompt)
        return result.content


def save_results(qi: int,
                 prompt: str,
                 answer: str,
                 rag_source_docs: list = None):
    """
    Save the result of LLM query.
    """
    rag_source_docs = rag_source_docs or []

    os.makedirs(output_dir, exist_ok=True)
    print(prompt, file=open(os.path.join(output_dir, f'question-{qi}.md'), 'w', encoding='utf-8'))
    print(answer, file=open(os.path.join(output_dir, f'answer-{qi}.md'), 'w', encoding='utf-8'))
    contexts_path = os.path.join(output_dir, f"rag_sources-{qi}.md")
    contexts_components = []
    for di, doc in enumerate(rag_source_docs):
        source = ''
        metadata = doc.metadata
        if 'source' in metadata:
            source = metadata['source']
        if not source:
            source = 'Unknown'
        source_txt = f'# Source {di + 1}/{len(rag_source_docs)}: '
        page_content = doc.page_content
        lines = page_content.splitlines()
        for li in range(len(lines)):
            line = lines[li]
            # remove title lines in the original md content
            if line.startswith('#') and line.find(' ') != -1:
                lines[li] = line[line.find(' ') + 1:]
        page_content = '\n'.join(lines)
        text_bytes = source_txt.encode() + source.encode() + b'\n' + page_content.encode('utf-8')
        contexts_components.append(text_bytes)
    contexts = b'\n\n'.join(contexts_components)
    open(contexts_path, 'wb').write(contexts)


def generate_report():
    file_list = [os.path.join(source_file_dir, file) for file in os.listdir(source_file_dir)]

    # check and generate markdown of source files
    print('Generating markdown files...')
    source_file_list, md_file_list = add_data_md(file_list)

    # check and generate the vector database
    print('Generating vector database...')
    embed_model = FlagAutoModel.from_finetuned(
        'BAAI/bge-m3',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=True,
        use_safetensors=True
    )
    add_vector_db(md_file_list, source_file_list, model=embed_model)

    print('Loading vector store...')
    persistent_data = load_persistent_data()

    print('Generating report...')

    md_files = os.listdir(source_file_md_dir)

    # iterate over the questions
    for qi, question in enumerate(report_config.questions):
        print(f'\nProcessing question {qi+1}/{len(report_config.questions)}: "{question.title}"...')

        # get source_file_list according to source_list of the question
        source_file_list = []
        for file in md_files:
            for source in question.source_list:
                if source in file:
                    match = True
                    break
            else:
                match = False
            if match:
                source_file_list.append(file)

        source_file_list = [os.path.join(source_file_md_dir, x) for x in source_file_list]
        if not source_file_list:
            print('WARNING: specified source_file_list is empty, skip this question')
            continue

        # construct the context which contains contents of relevant documents
        # determine retrieve input
        if question.source_hint:
            retrieve_input = question.source_hint
        else:
            retrieve_input = question.description
        rag_source_docs = construct_context(retrieve_input, persistent_data.vector_db, embed_model, source_file_list)

        # construct prompt
        # (1) source documents
        prompt_components = []
        max_doc_len = 8192
        for di, doc in enumerate(rag_source_docs):
            # truncate overlong documents
            page_content = doc.page_content
            if len(page_content) > max_doc_len:
                page_content = page_content[:max_doc_len] + '...'
            comp = f'源文档片段 {di+1}/{len(rag_source_docs)}：\n```\n{page_content}\n```'
            prompt_components.append(comp)
        # (2) main part of the question
        prompt_head = report_config.prompt_head
        prompt_components += [prompt_head, question.description]
        prompt = '\n\n'.join(prompt_components)

        # LLM query
        # answer = llm_run(prompt, 'qwen-plus')
        answer = llm_run(prompt, 'qwq-32b')

        # paste tables and images
        result_content = paste_tables_and_images(
            answer, persistent_data.table_db, persistent_data.image_db
        )

        # save question, answer and contexts
        save_results(qi, prompt, result_content, rag_source_docs)

    print('Finished.')


if __name__ == '__main__':
    generate_report()
