from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
import os
import shutil
from dotenv import load_dotenv
import ast

load_dotenv()  # take environment variables from .env.

def empty_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        return

    # Iterate over all files and directories within the specified directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)  # Get the full path of the item

        # Check if the item is a file or directory and delete accordingly
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove the file
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the directory and all its contents

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def string_to_dictionary(input_string):
    try:
        # Use ast.literal_eval to safely evaluate the string
        result = ast.literal_eval(input_string)
        if not isinstance(result, dict):
            raise ValueError("Input string does not represent a dictionary")
        return result
    except (ValueError, SyntaxError) as e:
        # Handle errors if the string is not a valid dictionary
        print(f"Error converting string to dictionary: {e}")
        return None

def pass_to_openai_vision_api_llama_index(image):
    gpt_prompt='''Above is the text extracted from an invoice.
You are an assistant tasked with extracting information from the invoice. Do this step by step.
1. First extract the date and due date.
2. Then assign it a category (e.g Food).
3. Extract the invoice number and vendor account number.
4. Extract the total amount.
5. Extract the items along with their name, quantity and individual price.
Output should only contain a dictionary in the following format
{
    "Date": "None",
    "Due Date": "None",
    "Category": "None",
    "Invoice Number": "None",
    "Vendor Account Number": "None",
    "Total Amount": "None",
    "Items": [
        {
            "Item": "None",
            "Quantity": "None",
            "Individual Price": "None"
        }
    ]
}
If a key is not mentioned in invoice or you dont understand, then make its value None
'''

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview", api_key=os.environ["OPENAI_API_KEY"], max_new_tokens=300
    )

    image_documents = SimpleDirectoryReader(input_files=[f"data/{image}"]).load_data()

    response = openai_mm_llm.complete(
        prompt=gpt_prompt,
        image_documents=image_documents,
    )

    response_dict = string_to_dictionary(response.text)

    return response_dict