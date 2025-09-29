import asyncio

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

from zc_agent import search_agent 
from zc_agent import load_data
from zc_agent import logs


def prettier_code_blocks():
    """Make rich code blocks prettier and easier to copy.

    From https://github.com/samuelcolvin/aicli/blob/v0.8.0/samuelcolvin_aicli.py#L22
    """
    class SimpleCodeBlock(CodeBlock):
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            code = str(self.text).rstrip()
            yield Text(self.lexer_name, style='dim')
            yield Syntax(
                code,
                self.lexer_name,
                theme=self.theme,
                background_color='default',
                word_wrap=True,
            )
            yield Text(f'/{self.lexer_name}', style='dim')

    Markdown.elements['fence'] = SimpleCodeBlock


def initialize_index():
    print("Initializing data ingestion...")
    index = load_data.index_data()
    print("Data indexing completed successfully!")
    return index


def initialize_agent(index):
    print("Initializing search agent...")
    agent = search_agent.init_agent(index)
    print("Agent initialized successfully!")
    return agent


async def main():
    prettier_code_blocks()
    console = Console()
    
    console.log("Starting the AI Assistant for DE Zoomcamp...", style='cyan')
    index = initialize_index()
    agent = initialize_agent(index)
    console.log("Ready to answer your questions!", style='green')
    console.log("Type 'stop' to exit the program.", style='yellow')

    while True:
        question = input("\nYour question: ")
        if question.strip().lower() in ['stop', 'exit', 'quit']:
            console.log("Goodbye!", style='cyan')
            break

        console.log(f"Processing: {question}...", style='cyan')
        
        with Live('', console=console, vertical_overflow='visible') as live:
            async with agent.run_stream(user_prompt=question) as result:
                async for message in result.stream_output():
                    live.update(Markdown(message))
        
        # Log the interaction
        logs.log_interaction_to_file(agent, result.new_messages())
        
        console.log("\n" + "="*50, style='dim')


if __name__ == "__main__":
    asyncio.run(main())
