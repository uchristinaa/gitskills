from run import llm_run_reasoning, llm_run

import pypandoc
import yaml
from collections.abc import Mapping, Sequence


# path settings
report_dir = 'report/CDE统计报告.docx' 
stat_yml_dir = 'configs/ich_e9.yml'
answer_stat_dir = 'output_ich_e9/answer_stat_v2.md'


def format_value(value):
    if value is None:
        return "`null`"
    elif isinstance(value, bool):
        return "`true`" if value else "`false`"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value
    else:
        return str(value)


def yaml_to_markdown(data, level=1, indent=0):
    """
    参数:
        data: YAML 解析后的 Python 对象
        level: 当前标题层级 (1-6)
        indent: 当前缩进级别
    """
    markdown = []
    indent_spaces = '  ' * indent  
    
    if isinstance(data, Mapping):  # 处理字典/映射类型
        for key, value in data.items():
            # 处理键
            if level <= 6:  
                header = f"{'#' * level} {key}"
            else: 
                header = f"**{key}**"
            
            # 处理值
            if isinstance(value, (Mapping, Sequence)) and not isinstance(value, str):
                markdown.append(f"{indent_spaces}{header}")
                markdown.append(yaml_to_markdown(value, level + 1, indent))
            else:
                markdown.append(f"{indent_spaces}{header}: {format_value(value)}")
    
    elif isinstance(data, Sequence) and not isinstance(data, str):  # 处理列表/序列类型
        for i, item in enumerate(data, 1):
            if isinstance(item, (Mapping, Sequence)) and not isinstance(item, str):
                item_header = f"{indent_spaces}{'#' * level} Item {i}" if level <= 6 else f"{indent_spaces}**Item {i}**"
                markdown.append(item_header)
                markdown.append(yaml_to_markdown(item, level + 1, indent))
            else:
                markdown.append(f"{indent_spaces}- {format_value(item)}")

    else:  # 处理基本类型
        markdown.append(f"{indent_spaces}{format_value(data)}")
    
    return "\n".join(markdown)


def generate_stat_report(stat_yml_dir, report_dir, answer_stat_dir):
    with open(stat_yml_dir, 'r') as file:
            config = yaml.safe_load(file)

    #构建prompt结构
    prompt_head = '输入报告：'
    prompt_report = pypandoc.convert_file(report_dir, 'md', extra_args=['--wrap=none']) 
    prompt_main = config['PromptMain']
    prompt_statstandards = yaml_to_markdown(config['StatStandards'])
    prompt = f"{prompt_head}\n{prompt_report}\n\n{prompt_main}\n\n{prompt_statstandards}"
    answer_stat = llm_run(prompt, model='qwen-plus')

    with open(answer_stat_dir, 'w', encoding='utf-8') as file:
        file.write(answer_stat)
    print(f"[Markdown] Saved: {answer_stat_dir}")


if __name__ == "__main__":
    generate_stat_report(stat_yml_dir, report_dir, answer_stat_dir)