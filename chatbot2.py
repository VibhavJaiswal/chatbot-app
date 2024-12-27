import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
from openai import OpenAI

# Set up OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")  # Retrieve API key from environment variable
if not api_key:
    raise ValueError("API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

client = OpenAI(api_key=api_key)

# Track visited URLs
visited_urls = set()

def fetch_website_content(base_url, max_depth=5):
    """
    Recursively fetch content from a website up to a certain depth.
    """
    if max_depth == 0 or base_url in visited_urls:
        return ""

    visited_urls.add(base_url)

    try:
        # Fetch page content
        response = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content
        page_content = soup.get_text(separator="\n", strip=True)

        # Find internal links
        links = [
            urljoin(base_url, link.get('href'))
            for link in soup.find_all('a', href=True)
            if link.get('href').startswith('/') and urljoin(base_url, link.get('href')) not in visited_urls
        ]

        # Recursively fetch content from linked pages
        for link in links:
            page_content += "\n" + fetch_website_content(link, max_depth - 1)

        return page_content
    except Exception as e:
        print(f"Failed to fetch {base_url}: {e}")
        return ""

# Starting URL
base_url = "https://optistaff.in/"
print("Fetching website content...")
website_content = fetch_website_content(base_url, max_depth=5)  # Increased max_depth for maximum information

# Chunk text for embedding
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

documents = chunk_text(website_content)

# Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = FAISS.from_texts(documents, embeddings)

# Define query function with improved context handling
def get_answer(query):
    try:
        # Retrieve top 5 relevant documents for the query
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        if not context.strip():
            return "No relevant information found on the website for your query."

        # Use OpenAI's chat completions API
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the query accurately."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.7,
            max_tokens=1500,  # Increased token limit for more detailed responses
        )
        return completion.choices[0].message.content.strip()
    except client.errors.AuthenticationError:
        return "Authentication error: Please check your OpenAI API key."
    except client.errors.RateLimitError:
        return "Rate limit error: Too many requests. Try again later."
    except client.errors.BadRequestError as e:
        return f"Invalid request: {e}"
    except client.errors.APIError as e:
        return f"OpenAI API error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Build Gradio interface with Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Website Data Assistant")
    
    # Chatbot inputs and outputs
    with gr.Row():
        query_input = gr.Textbox(label="Enter your query", placeholder="Type your question here...", lines=4)
        response_output = gr.Textbox(label="Response")
    
    # Buttons for submit
    submit_button = gr.Button("Submit")
    
    # Bind function to button
    submit_button.click(fn=get_answer, inputs=query_input, outputs=response_output)

# Get the PORT from environment variables (default to 8080 if not set)
port = int(os.environ.get("PORT", 8080))

# Launch the Gradio app with correct server settings
demo.launch(server_name="0.0.0.0", server_port=port)
