from LLM_client import ConversationSession  # or wherever you land ConversationSession

def run_cli():
    try:
        print("=" * 60)
        print("FITNESS & WELLNESS RAG CHATBOT")
        print("=" * 60)
        print("Type your questions below. Type 'exit' or 'quit' to end.\n")

        session = ConversationSession()

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break
            if not user_input:
                print("Please enter a question.\n")
                continue
            try:
                answer, chunks, retrieval_time, generation_time = session.ask(user_input)
                print(f"\nAssistant: {answer}\n")

                printedChunks=f"Chunks:\n"+ "\n\n".join(str(chunk) for chunk in chunks)
                print (printedChunks)
                # print(f"Chunks:\n"+ "\n\n".join(str(chunk) for chunk in chunks))
                print(f"Retrieval Time: {retrieval_time:.2f} seconds")
                print(f"Generation Time: {generation_time:.2f} seconds\n")
            except Exception as e:
                print(f"\nError: {e}\n")
    except Exception as e:
            print(f"RAG pipeline failed: {e}")
            exit(1)


if __name__ == "__main__":
    run_cli()