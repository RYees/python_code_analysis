import pandas as pd
import numpy as np
import re

def read_data(file_path):
    data = pd.read_csv(file_path)
    data = pd.DataFrame(data)
    return data

def drop_column(df):
    # Drop a column
    columns_to_drop = ['submission_id','payload', 'rubric_id', 'rubric_type', 'next_action', 'request_origin', 'params']
    df = df.drop(columns_to_drop, axis=1)
    print("DataFrame after dropping unwanted column")
    return df

def clean_packages(text):
    patterns = [
    r"### BEGIN File: \w+\.txt ###.*?END File: \w+\.txt ###",
    r"### BEGIN File: (\w+/\w+.txt) ###.*?END File: \1 ###",
    r"### BEGIN File: (\w+/\w+.txt) ###.*?END File: (\w+/\w+.txt) ###",
    r"### BEGIN File: ([\w./-]+\.txt) ###.*?END File: ([\w./-]+\.txt) ###",
    r"### BEGIN File: (\w+/\w+.pgerd) ###.*?END File: (\w+/\w+.pgerd) ###",
    r"### BEGIN File: \w+\.lock ###.*?END File: \w+\.lock ###",
    r"### BEGIN File: \w+\.yml ###.*?END File: \w+\.yml ###",
    r"### BEGIN File: \w+\.yaml ###.*?END File: \w+\.yaml ###",
    r"### BEGIN File: \w+\.toml ###.*?END File: \w+\.toml ###",
    r"### BEGIN File: \w+\.log ###.*?END File: \w+\.log ###",
    #r"### BEGIN File: model_schema\/model_schema_design\.pgerd ###.*?END File: model_schema\/model_schema_design\.pgerd ###",
    r"### BEGIN File: package\.json ###.*?END File: package\.json ###",
    r"### BEGIN File: setup\.py ###.*?END File: setup\.py ###",
    r"### BEGIN File: tailwind.config\.js ###.*?END File: tailwind.config\.js ###",
    r"### BEGIN File: tailwind.config\.ts ###.*?END File: tailwind.config\.ts ###",
    r"### BEGIN File: Dockerfile ###.*?END File: Dockerfile ###",
    ]
    cleaned_text = re.sub("|".join(patterns), '', text, flags=re.DOTALL | re.MULTILINE)
    return cleaned_text
#textt = clean_packages(data['code_content'][151])
# week6_rag['code_content'] = week6_rag['code_content'].apply(clean_packages)

def clean_content(text):
    pattern = r"### BEGIN File: README\.md ###.*?END File: README\.md ###"
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text
#data['code_content'][182] = clean_content(data['code_content'][182])
# Assuming your DataFrame is named 'data' and the column containing the text is named 'code_content'
# week6_rag['code_content'] = week6_rag['code_content'].apply(clean_content)

def clean_License(text):
    pattern = r'BEGIN File: LICENSE.*?END File: LICENSE'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text
# data['code_content'][182] = clean_License(data['code_content'][182])
# Assuming your DataFrame is named 'data' and the column containing the text is named 'code_content'
# week6_rag['code_content'] = week6_rag['code_content'].apply(clean_License)

def remove_comments(text):
    pattern = r'#.*?\n'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text
# data['code_content'][182] = remove_comments(data['code_content'][182])
# Assuming your DataFrame is named 'data' and the column containing the text is named 'code_content'
# week6_rag['code_content'] = week6_rag['code_content'].apply(remove_comments)

def clean_perged_files(text):
    pattern = r'### BEGIN File:.*?model_schema/model_schema_design\.pgerd.*?### END File'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text
# data['code_content'][182] = clean_perged_files(data['code_content'][182])
# Assuming your DataFrame is named 'data' and the column containing the text is named 'code_content'
# week6_rag['code_content'] = week6_rag['code_content'].apply(clean_perged_files)

def extract_content(text):
    pattern = r'Content:\s*(.*?)(?=\s*Commit History:)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return '\n'.join(matches)
    else:
        return text
# data['code_content'][182] = extract_content(data['code_content'][182])
# Assuming your DataFrame is named 'df' and the column containing the text is named 'text_column'
# week6_rag['code_content'] = week6_rag['code_content'].apply(extract_content)

def clean_installs(text):
    pattern = r"pip install \w+\n"
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text
# data['code_content'][182] = clean_perged_files(data['code_content'][182])
# Assuming your DataFrame is named 'data' and the column containing the text is named 'code_content'
# week6_rag['code_content'] = week6_rag['code_content'].apply(clean_installs)

def remove_extra_spaces(strings):
    updated_strings = []
    pattern = r"(?<!\S)\s+(?!\S)"

    for string in strings:
        updated_string = re.sub(pattern, " ", string)
        updated_strings.append(updated_string)

    return updated_strings

def remove_extra_newlines(strings):
    updated_strings = []
    pattern = r"\n+"

    for string in strings:
        updated_string = re.sub(pattern, "\n", string.strip())
        updated_strings.append(updated_string)

    return updated_strings

def extract(text):
    pattern = r'Content:\s*(.*?)$'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return '\n'.join(matches)
    else:
        return text

def remove_emptycontent_rows(df):
    # Iterate over the rows
    for index, row in df.iterrows():
        if row['code_content'].startswith('Repository Structure:'):
            # Drop the row with the specified index
            df = df.drop(index)
    # Reset the index of the DataFrame
    data = df.reset_index(drop=True)
    return data

def checking_for_empty_value(df):
    column_name = 'code_content'

    # Check if the column contains empty strings
    has_empty_values = any(df[column_name] == '')

    if has_empty_values:
        return "The column contains empty string values."
    else:
        return "The column does not contain empty string values."
    