from crewai import Agent, Task, Crew, Process, LLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# LangChain + Ollama
langchain_llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.3
)

# CrewAI + Ollama
crewai_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

HISTORY_FILE = "health_history.txt"


def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "No previous health history found."


def save_history(symptoms, response):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write("\n\n--- New Health Record ---\n")
        f.write(f"Symptoms: {symptoms}\n")
        f.write(f"Response: {response}\n")


def langchain_pre_check(symptoms):
    prompt = ChatPromptTemplate.from_template("""
    You are a healthcare safety assistant.

    User symptoms: {symptoms}

    First check if symptoms look urgent.
    Give only:
    1. Emergency risk: Low / Medium / High
    2. Main symptoms found
    3. Short safety note

    Do not give final diagnosis.
    """)

    chain = prompt | langchain_llm
    result = chain.invoke({"symptoms": symptoms})
    return result.content


def run_healthcare_bot(symptoms):
    history = load_history()
    safety_check = langchain_pre_check(symptoms)

    symptom_analyzer = Agent(
        role="Symptom Analyzer",
        goal="Analyze symptoms and identify possible general conditions.",
        backstory="You analyze symptoms safely and never provide final diagnosis.",
        llm=crewai_llm,
        verbose=True
    )

    medical_agent = Agent(
        role="Medical Knowledge Agent",
        goal="Provide general medical information and basic health education.",
        backstory="You explain health topics in simple language.",
        llm=crewai_llm,
        verbose=True
    )

    recommendation_agent = Agent(
        role="Recommendation Agent",
        goal="Suggest basic remedies and recommend doctor consultation when needed.",
        backstory="You give safe health suggestions and emergency warning signs.",
        llm=crewai_llm,
        verbose=True
    )

    monitoring_agent = Agent(
        role="Monitoring Agent",
        goal="Track health history and repeated symptoms.",
        backstory="You monitor previous health records and summarize patterns.",
        llm=crewai_llm,
        verbose=True
    )

    tasks = [
        Task(
            description=f"""
            Symptoms: {symptoms}
            LangChain safety check: {safety_check}

            Analyze symptoms and mention possible general conditions.
            Do not give final diagnosis.
            """,
            expected_output="Possible general conditions.",
            agent=symptom_analyzer
        ),

        Task(
            description=f"""
            Symptoms: {symptoms}

            Give general health information, possible causes, and prevention tips.
            Keep language simple.
            """,
            expected_output="General medical information.",
            agent=medical_agent
        ),

        Task(
            description=f"""
            Symptoms: {symptoms}
            Safety check: {safety_check}

            Suggest basic remedies.
            Recommend doctor consultation if symptoms are serious.
            Mention emergency warning signs.
            Do not prescribe medicine dosage.
            """,
            expected_output="Basic remedies and consultation advice.",
            agent=recommendation_agent
        ),

        Task(
            description=f"""
            Current symptoms: {symptoms}
            Previous health history: {history}

            Track and summarize health history.
            Mention repeated symptoms if found.
            """,
            expected_output="Health monitoring summary.",
            agent=monitoring_agent
        )
    ]

    crew = Crew(
        agents=[
            symptom_analyzer,
            medical_agent,
            recommendation_agent,
            monitoring_agent
        ],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    crew_result = crew.kickoff()

    final_response = f"""
LANGCHAIN SAFETY CHECK:
{safety_check}

CREWAI HEALTH GUIDANCE:
{crew_result}

Important: This chatbot provides general health information only.
It is not a replacement for a doctor.
"""

    save_history(symptoms, final_response)
    return final_response


print("===================================")
print(" Healthcare Assistant Chatbot")
print(" LangChain + CrewAI + Ollama")
print("===================================")
print("Type 'exit' to stop.\n")

while True:
    symptoms = input("Enter your symptoms: ")

    if symptoms.lower() == "exit":
        print("Thank you. Stay healthy!")
        break

    if symptoms.strip() == "":
        print("Please enter symptoms.\n")
        continue

    response = run_healthcare_bot(symptoms)

    print("\nAI Health Guidance:")
    print(response)
    print("\n-----------------------------------\n")