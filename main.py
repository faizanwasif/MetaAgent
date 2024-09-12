import os
import anthropic
from collections import namedtuple
import json
import random
from dotenv import load_dotenv
import backoff
from typing import Union
from datetime import datetime
import sys
import io
from contextlib import redirect_stdout
import argparse
from duckduckgo_search import DDGS

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv('ANTHROPIC_API_KEY')

# Named tuple for holding information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for FM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\n"

# Description of the role of the FM Module
ROLE_DESC = lambda role: f"You are a {role}."

class FMModule:
    def __init__(self, output_fields, name, role='helpful assistant', model='claude-3-haiku-20240307', temperature=0.5):
        self.output_fields = output_fields
        self.name = name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = f"{name}_{random.randint(1000, 9999)}"
        
        # Initialize the Anthropic client with the API key
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_prompt(self, input_infos, instruction):
        system_prompt = ROLE_DESC(self.role)
        prompt = FORMAT_INST(self.output_fields)
        prompt += f"# Your Task:\n{instruction}\n\n"
        for info in input_infos:
            prompt += f"### {info.name} by {info.author}:\n{info.content}\n\n"
        return system_prompt, prompt

    @backoff.on_exception(backoff.expo, anthropic.RateLimitError)
    def query(self, input_infos, instruction, iteration_idx=-1):
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.content[0].text
        
        try:
            parsed_content = json.loads(content)
            return [Info(field, self.id, parsed_content.get(field, ''), iteration_idx) for field in self.output_fields]
        except json.JSONDecodeError:
            print(f"Failed to parse JSON. Raw content: {content}")
            return [Info(field, self.id, '', iteration_idx) for field in self.output_fields]

    def __repr__(self):
        return f"{self.name}_{self.id}"

    def __call__(self, input_infos, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

def search_web(query, num_results=5):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return [{"title": r['title'], "snippet": r['body']} for r in results]

class MetaAgentSearch:
    def __init__(self, domain_description, framework_code, max_iterations=25):
        self.domain_description = domain_description
        self.framework_code = framework_code
        self.max_iterations = max_iterations
        self.archive = []
        self.meta_agent = FMModule(['thought', 'name', 'code', 'tools'], 'MetaAgent', model='claude-3-5-sonnet-20240620', temperature=0.8)

    def generate_main_prompt(self):
        tools_query = f"Python libraries and tools for {self.domain_description}"
        search_results = search_web(tools_query)
        tools_info = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results])

        prompt = f"""You are an expert machine learning researcher designing agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on {self.domain_description}

Framework Code Example:
{self.framework_code}

Relevant Tools and Libraries:
{tools_info}

# Your task
You are deeply familiar with prompting techniques and the agent works from the literature. Your goal is to maximize the specified performance metrics by proposing interestingly new agents.

Observe the discovered agents carefully and think about what insights, lessons, or stepping stones can be learned from them.

Be creative when thinking about the next interesting agent to try. You are encouraged to draw inspiration from related agent papers or academic literature from other research areas.

Use the knowledge from the archive, inspiration from academic literature, and the provided tools and libraries to propose the next interesting agentic system design.

THINK OUTSIDE THE BOX.

Discovered Agent Archive:
{self.format_archive()}

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what the next interesting agent to try should be, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture.
The third key ("code") corresponds to the exact "forward()" function in Python code that you would like to try. You must write COMPLETE CODE in "code".
The fourth key ("tools") should list the tools and libraries you propose to use for this agent.

Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent:
{{
"thought": "**Insights:** Your insights on what should be the next interesting agent. **Overall Idea:** your reasoning and the overall concept behind the agent design. **Implementation:** describe the implementation step by step.",
"name": "Name of your proposed agent",
"code": "def forward(self, taskInfo): # Your code here",
"tools": ["Tool1", "Tool2", "Tool3"]
}}
"""
        return prompt

    def format_archive(self):
        return "\n".join([f"Agent {i+1}: {agent['name']}\nPerformance: {agent['performance']}\nCode:\n{agent['code']}\nTools: {agent['tools']}\n" 
                          for i, agent in enumerate(self.archive)])

    def reflect_and_improve(self, proposal):
        reflection_prompt = f"""
    {json.dumps(proposal)}

    Carefully review the proposed new architecture and reflect on the following points:

    1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings.

    2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version.

    3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.

    4. **Tool Usage**: Evaluate whether the proposed tools are appropriate for the task and if there are any other tools that could be beneficial.

    Your response should be organized as follows:
    "reflection": Provide your thoughts on the above points and suggest improvements.
    "thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.
    "name": Provide a name for the revised or new architecture. (Don't use words like "new" or "improved" in the name.)
    "code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
    "tools": Provide an updated list of tools and libraries to be used.
    """
        reflection = self.meta_agent.query([Info('reflection_prompt', 'MetaAgentSearch', reflection_prompt, -1)], 
                                        "Reflect and improve the proposed agent.")[0]
        
        # Ensure reflection.content is a string (join if it's a list)
        reflection_content = reflection.content
        if isinstance(reflection_content, list):
            reflection_content = "".join(reflection_content)

        try:
            return json.loads(reflection_content)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from reflection. Raw content: {reflection_content}")
            return {
                "reflection": "Failed to parse reflection",
                "thought": "Error in reflection process",
                "name": "Error in agent naming",
                "code": "# Error in code generation",
                "tools": []
            }


    def evaluate_agent(self, agent_code):
        # Placeholder for evaluation function
        # In a real implementation, this would run the agent on a validation set and return performance metrics
        return random.random()

    def search(self):
        for i in range(self.max_iterations):
            main_prompt = self.generate_main_prompt()
            proposal = self.meta_agent.query([Info('main_prompt', 'MetaAgentSearch', main_prompt, i)], 
                                             "Propose the next interesting agent.")[0]
            proposal_content = proposal.content
            
            # Reflection and improvement
            improved_proposal = self.reflect_and_improve(proposal_content)
            
            # Evaluate the agent
            performance = self.evaluate_agent(improved_proposal['code'])
            
            # Add to archive
            self.archive.append({
                'name': improved_proposal['name'],
                'code': improved_proposal['code'],
                'tools': improved_proposal['tools'],
                'performance': performance
            })
            
            print(f"Iteration {i+1}: Agent '{improved_proposal['name']}' - Performance: {performance}")
            print(f"Tools: {', '.join(improved_proposal['tools'])}")

        # Return the best agent from the archive
        best_agent = max(self.archive, key=lambda x: x['performance'])
        return best_agent

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run Meta Agent Search with specified parameters.")
    parser.add_argument("--domain", type=str, default="Article Findings on Web",
                        help="The domain for which to create an agent")
    parser.add_argument("--iterations", type=int, default=25,
                        help="The number of iterations to run the search")
    args = parser.parse_args()

    # Use the parsed arguments
    domain_description = args.domain
    max_iterations = args.iterations

    framework_code = """
    from crewai import Agent, Task, Crew
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    import os

    class CEOAgent:
        def __init__(self, api_key):
            load_dotenv()  # Load environment variables
            self.model = ChatOpenAI(temperature=0, api_key=api_key)  # Initialize language model
            self.ceo_agent = self.create_ceo_agent()  # Create the CEO agent
            self.decision_making_task = self.create_task()  # Create the decision-making task
            self.crew = self.create_crew()  # Create the crew

        def create_ceo_agent(self):
            \"\"\"Creates the CEO agent with role, goal, and backstory.\"\"\"
            return Agent(
                role="CEO",
                goal="Provide informed decisions and guidance on any question related to the company's operations, strategy, and vision in the AI industry.",
                backstory=(
                    "You are the CEO of an innovative AI company that specializes in developing cutting-edge artificial intelligence solutions. "
                    "Your expertise spans AI technologies, business strategy, leadership, and industry trends. "
                    "You are responsible for making strategic decisions that drive the company's growth and success."
                ),
                allow_delegation=False,
                verbose=True
            )

        def create_task(self):
            \"\"\"Creates the decision-making task for the CEO.\"\"\"
            return Task(
                description=(
                    "1. Analyze the following question: '{question}'.\\n"
                    "2. Provide a clear and decisive answer, considering the company's goals, resources, and market position.\\n"
                    "3. Justify your decision with logical reasoning and reference to industry best practices or company policies.\\n"
                    "4. Offer any additional recommendations that could benefit the company's strategic objectives."
                ),
                expected_output=(
                    "A well-reasoned decision or guidance addressing the question, demonstrating strategic thinking and leadership."
                ),
                agent=self.ceo_agent
            )

        def create_crew(self):
            \"\"\"Creates the crew with the CEO agent and task.\"\"\"
            return Crew(
                agents=[self.ceo_agent],
                tasks=[self.decision_making_task],
                verbose=2
            )

        def kickoff(self, question_input):
            \"\"\"Kicks off the crew with a given question and returns the result.\"\"\"
            return self.crew.kickoff(inputs=question_input)
    """

    # Capture console output
    console_output = io.StringIO()
    with redirect_stdout(console_output):
        meta_agent_search = MetaAgentSearch(domain_description, framework_code, max_iterations=max_iterations)
        best_agent = meta_agent_search.search()
        
        print(f"\nBest Agent for {domain_description}:")
        print(f"Name: {best_agent['name']}")
        print(f"Performance: {best_agent['performance']}")
        print("Code:")
        print(best_agent['code'])
        print("Tools:")
        print(", ".join(best_agent['tools']))

    # Get the captured output
    output = console_output.getvalue()

    # Write the output to a file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"agent_code_{timestamp}.txt"
    
    with open(filename, "w") as file:
        file.write(output)

    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()