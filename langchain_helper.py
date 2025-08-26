from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = OpenAI(temperature=0.5)

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template=(
            "You are an expert pet namer. "
            "I have a {animal_type} pet and I want a cool name for it. "
            "it is {pet_color} in color. "
            "Suggest me five cool names for my pet."
        )
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")
    response = name_chain.invoke({'animal_type': animal_type, 'pet_color': pet_color})
    return response

def langchain_agent():
    llm = OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    result = agent.invoke(
        "Look up the average life expectancy of a dog in years, then multiply that number by 3. Return only the final number."
    )

    print(result)

if __name__ == "__main__":
    langchain_agent()
    print(generate_pet_name("snake", "azure"))