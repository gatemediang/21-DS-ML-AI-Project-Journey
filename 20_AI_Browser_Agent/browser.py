# This is the link to the video explanation of this code with the result copied as json on the CLI
# : https://www.loom.com/share/f7d297cddcc44d29b57d62abf518e53b

import os
from dotenv import load_dotenv
from browser_use import Agent, ChatGoogle
import asyncio

# Load .env from parent directory
load_dotenv(dotenv_path="../.env")

# Access the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


async def main():
    llm = ChatGoogle(model="gemini-2.5-flash")
    # task = "Find the number 1 post on Show HN"
    task = "Search Google for 'what is browser automation' and summarize the top result in three results in a json file."
    agent = Agent(task=task, llm=llm)
    history = await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
