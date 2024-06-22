import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# Instantiate the Ollama model
ollama_openhermes = Ollama(model="openhermes")

st.title("AI Blog Writer")

# User input for the blog topic
blog_topic = st.text_input("Enter the topic for the blog post:")

if blog_topic:
    # Define the agents with roles, goals, and backstory
    researcher = Agent(
        role='Senior Research Analyst',
        goal=f"""Uncover recent developments in {blog_topic} in 2024""",
        backstory="""You work at a leading think tank.
        Your expertise lies in identifying emerging trends, predicting future problems and solutions.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=True,
        allow_delegation=False,
        llm=ollama_openhermes
    )

    editor = Agent(
        role='Editor',
        goal='Polish and refine the content for readability and engagement , keeping in my mind that the content should not be decreased',
        backstory=f"""You are a seasoned editor with years of experience in  journalism.
        You ensure the content is error-free and engaging, maintaining the original essence of the content.""",
        verbose=True,
        allow_delegation=False,
        llm=ollama_openhermes
    )

   



    # Define the tasks with the user-provided topic
    task1 = Task(
        description=f"""Conduct a comprehensive analysis  related to {blog_topic} in present and foreseen future .
        Focus on specific information which has potential to become big in future and give money to the end user 
        Provide a detailed analysis of your information in easy and crisp manner .""",
        expected_output=f"Detailed report with key insights which is important  for {blog_topic} in 2024",
        agent=researcher    
    )

    task2 = Task(
        description="Review the researcher's report, correct any grammatical errors, and improve the overall readability. Ensure the content flows well and maintains an engaging tone.",
        expected_output="A polished and error-free research report. of 2000 words ",
        agent=editor
    )





    # Instantiate the crew with a sequential process
    crew = Crew(
        agents=[researcher, editor,],
        tasks=[task1, task2, ],
        llm=ollama_openhermes,
        verbose=3,  # Adjust verbosity level for more detailed logs if needed
        process=Process.sequential
    )

    # Start the process and collect the results
    result = crew.kickoff()

    # Print only the final summary
    st.header("Blog Post Summary:")
    st.write(result)
