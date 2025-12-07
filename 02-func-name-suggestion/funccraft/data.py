from pathlib import Path

import datasets
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import re

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def extract_function_info(code: str):
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    def find_function(node):
        if node.type == "function_definition":
            return node
        for child in node.children:
            res = find_function(child)
            if res:
                return res
        return None

    func_node = find_function(root)
    if func_node is None:
        return None, None, None

    name_node = func_node.child_by_field_name("name")
    if name_node is None:
        return None, None, None

    func_name = code[name_node.start_byte:name_node.end_byte]

    block = func_node.child_by_field_name("body")
    if block is None:
        return func_name, "", ""

    block_text = code[block.start_byte:block.end_byte]
    body_with_comments = block_text
    body_no_comments = re.sub(r"[ \t]*#.*", "", block_text)

    body_no_comments = re.sub(
        r'^\s*(?P<quote>"""|\'\'\')(?:.|\n)*?\1',
        "",
        body_no_comments,
        count=1,
        flags=re.DOTALL
    )

    return func_name, body_no_comments.strip(), body_with_comments.strip()


def check_extraction(dataset, n=10):
    total_name_matches = 0
    total_doc_matches = 0
    total_rows = len(dataset)

    def extract_doc(body: str) -> str:
        if '"""' in body:
            parts = body.split('"""')
            if len(parts) > 1:
                return parts[1].strip()
        return ""

    for i, row in enumerate(dataset):
        original_name = row.get("func_name", "").split(".")[-1]
        original_doc = row.get("func_documentation_string", "").strip()

        extracted_name = row.get("extracted_name", "")
        extracted_doc = extract_doc(row.get("body_with_comments", ""))
        code = row.get("body_no_comments")

        name_match = original_name == extracted_name
        doc_match = original_doc == extracted_doc
        total_name_matches += name_match
        total_doc_matches += doc_match

        if i < n:
            print(f"Example {i+1}:")
            print(f"  Original name:   {original_name}")
            print(f"  Extracted name:  {extracted_name}")
            print(f"  Name match:      {name_match}")
            print(f"  Code without comms: {code[:200]}{'...' if len(code) > 200 else ''}")
            print(f"  Original doc:    {original_doc[:200]}{'...' if len(original_doc) > 200 else ''}")
            print(f"  Extracted doc:   {extracted_doc[:200]}{'...' if len(extracted_doc) > 200 else ''}")
            print(f"  Doc match:       {doc_match}")
            print("-" * 50)

    name_match_pct = total_name_matches / total_rows * 100
    doc_match_pct = total_doc_matches / total_rows * 100

    print("\nOverall dataset statistics:")
    print(f"  Total examples:        {total_rows}")
    print(f"  Name matches:          {total_name_matches} ({name_match_pct:.2f}%)")
    print(f"  Documentation matches: {total_doc_matches} ({doc_match_pct:.2f}%)")


def prepare() -> datasets.Dataset:
    print("Loading CodeSearchNet...")
    raw = datasets.load_dataset(
        "code_search_net",
        "python",
        split="test",
        trust_remote_code=True,
    )

    raw = raw.select(range(1000))

    new_fields = {
        "extracted_name": [],
        "body_no_comments": [],
        "body_with_comments": [],
    }

    print("Parsing functions")

    for row in raw:
        func_code = row["whole_func_string"]
        name, clean_body, full_body = extract_function_info(func_code)

        name = name or ""
        clean_body = clean_body or ""
        full_body = full_body or ""

        new_fields["extracted_name"].append(name)
        new_fields["body_no_comments"].append(clean_body)
        new_fields["body_with_comments"].append(full_body)

    print("Merging fields...")

    for key, value in new_fields.items():
        raw = raw.add_column(key, value)
    check_extraction(raw)

    return raw


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
