import sqlite3
import ollama

# 1. Set up SQLite database and store sample documents
def setup_database():
    conn = sqlite3.connect('myfile.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY,
                        content TEXT
                      )''')
    sample_documents = [
        ("The first letter is alpha.",),
        ("The second letter is beta.",)
    ]
    cursor.executemany("INSERT INTO documents (content) VALUES (?)", sample_documents)
    conn.commit()
    conn.close()

setup_database()

# 2. Function to retrieve documents based on query
def retrieve_documents(query):
    conn = sqlite3.connect('myfile.db')
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM documents WHERE content LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    conn.close()
    return [result[0] for result in results]

# 3. Function to generate a response using Ollama
def generate_response(query, documents):
    context = "\n".join(documents)
    prompt = f"Query: {query}\n\nContext:\n{context}\n\nResponse:"
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    return response['message']

# 4. Complete flow: Query input, document retrieval, and response generation
def handle_query(query):
    retrieved_docs = retrieve_documents(query)
    if retrieved_docs:
        response = generate_response(query, retrieved_docs)
        return response
    else:
        return "No relevant documents found."

# Test the whole process with a sample query
query = "first"
response = handle_query(query)
print(f"Response: {response}")