from typing import Any
import json
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


# ------------------------- Tools -------------------------
@tool(parse_docstring=True)
def generate_sample_data(data_fields: dict[str, list]) -> dict[str, Any]:
    """
    Generate sample data with custom fields and values.

    Examples:
        generate_sample_data(
            data_fields={
                "firstName": ["Alice", "Bob"],
                "lastName": ["Smith", "Johnson"],
                "age": [28, 34],
                "email": ["asmith@test.com", "bobjohnson@example.com"]
            }
        )

    Args:
        data_fields (dict[str, list]): A dictionary where each key is the 
            name of a field and each value is a list of values for that 
            field. All lists should be the same length. Otherwise, the tool 
            will generate only as many records as the shortest list allows.

    Returns:
        dict[str, Any]: Dictionary with 'data' list of dictionaries or 'error' message
    """

    # Validation
    if not all(isinstance(k, str) and isinstance(v, list) for k, v in data_fields.items()):
        return {"error": "Invalid input format. data_fields must be a dictionary of string keys and list values."}
    if not all(v for v in data_fields.values()):
        return {"error": "All lists in data_fields must be non-empty."}
    
    num_records = min(len(v) for v in data_fields.values())    
    data = []

    for i in range(num_records):
        data.append({k: v[i] for k, v in data_fields.items()})

    return {"data": data}


@tool(parse_docstring=True)
def write_json(data: list[dict[str, Any]], filepath: str) -> str:
    """
    Given a Python list of user data dictionaries, 
    write it as JSON to a file with pretty formatting.

    Args:
        data (list[dict[str, Any]]): List of user data dictionaries
        filepath (str): Path to the output JSON file

    Returns:
        str: Success or error message
    """

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Successfully wrote JSON data to '{filepath}' ({len(json.dumps(data))} characters)."
    
    except Exception as e:
        return f"Error writing JSON: {str(e)}"
    

@tool(parse_docstring=True)
def read_json(filepath: str) -> str:
    """
    Read and return the contents of a JSON file.

    Args:
        filepath (str): Path to the input JSON file

    Returns:
        str: Contents of the JSON file as a string or error message
    """

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file: {str(e)}"
    
    except Exception as e:
        return f"Error reading JSON: {str(e)}"
# ---------------------------------------------------------


model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3
)

tools = [generate_sample_data, write_json, read_json]

system_prompt = """
You are DataGen, a helpful assistant that generates sample data for applications. 
Ask the user for the fields they want and any specific values or constraints and 
the number of records needed before you generate any data. When you generate 
random data, make sure it is realistic. Always present the data to the user once 
you generate it, even if they didn't ask you to. Do not save the generated data to 
a JSON file unless the user asks you.
"""

agent = create_agent(model, tools, system_prompt=system_prompt)


def run_agent(user_input: str, history: list[BaseMessage]) -> AIMessage:
    """
    Single-turn agent runner with automatic tool execution.
    """
    
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )

        # Return AI's message (last message)
        return result["messages"][-1]
    
    except Exception as e:
        # Return error as an AI message so the conversation can continue
        return AIMessage(content=f"Error: {str(e)}\n\nPlease try rephrasing your request or provide more specific details.")
    

if __name__ == "__main__":
    print("=" * 100)
    print("DataGen Agent - Sample Data Generator")
    print("=" * 100)
    print("Generate sample data and save to JSON files.")
    print()
    print("Examples:")
    print("  - Generate 5 sample users with fields: name, age, city.")
    print("  - Make a dataset of 3 products with name, price, and category. Then save it to products.json.")
    print("  - Make 10 sample employees with fields: name, department, and salary. Salary should be between 4000 and 8000.")
    print("  - Create data with fields: first_names [Alice, Bob], last_names [Stone, Diaz], ages [22, 30]. Show the table.")
    print("  - Read the contents of users.json and display the data in a table.")
    print()
    print("Commands: 'quit' or 'exit' to end")
    print("=" * 100)

    history: list[BaseMessage] = []

    while True:
        user_input = input("You: ").strip()
        print("-" * 100)

        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "q", ""]:
            print("Goodbye!")
            break

        print("Agent: ", end="", flush=True)
        response = run_agent(user_input, history)
        print(response.content)
        print("-" * 100)

        # Update conversation history

        history += [HumanMessage(content=user_input), response]
