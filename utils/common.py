import json

# 读取 JSON 文件并返回其内容
def get_json_content(json_file_path: str):
    try:
        # 以只读模式（"r"）打开指定路径的 JSON 文件
        with open(json_file_path, "r") as f:
            # 将文件内容解析为 Python 对象（通常是字典或列表）
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error reading JSON file '{json_file_path}': {e}!")
