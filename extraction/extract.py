from tika import parser

def extract_text_from_file(file_path):
    parsed = parser.from_file(file_path)
    return parsed['content']

