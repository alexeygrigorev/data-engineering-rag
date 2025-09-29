import ingest
import search_agent 
import logs

import asyncio


REPO_OWNER = "DataTalksClub"
REPO_NAME = "faq"


def initialize_index():
    print(f"Starting AI FAQ Assistant for {REPO_OWNER}/{REPO_NAME}")
    print("Initializing data ingestion...")

    def filter(doc):
        return 'data-engineering' in doc['filename']

    index = ingest.index_data(REPO_OWNER, REPO_NAME, filter=filter)
    print("Data indexing completed successfully!")
    return index


def initialize_agent(index):
    print("Initializing search agent...")
    agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME)
    print("Agent initialized successfully!")
    return agent


def main():
    index = initialize_index()
    agent = initialize_agent(index)
    print("\nReady to answer your questions!")
    print("Type 'stop' to exit the program.\n")

    while True:
        question = input("Your question: ")
        if question.strip().lower() == 'stop':
            print("Goodbye!")
            break

        print("Processing your question...")
        response = asyncio.run(agent.run(user_prompt=question))
        logs.log_interaction_to_file(agent, response.new_messages())

        print("\nResponse:\n", response.output)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
