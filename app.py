from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

print("🎓 Student AI Doubt Solver Chatbot")
print("Ask doubts about DBMS, OS, Java, Python, DSA, AI")
print("Type 'exit' to stop\n")

while True:
    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("Chatbot stopped.")
        break

    prompt = f"""
You are a Student AI Doubt Solver Chatbot.

Rules:
- Explain in simple words
- Give step-by-step answer
- Give example if needed
- Use easy language for MCA students
- If question is coding related, give clean code also

Student Question:
{user_question}
"""

    response = llm.invoke(prompt)

    print("\nBot:", response.content)
    print("-" * 50)