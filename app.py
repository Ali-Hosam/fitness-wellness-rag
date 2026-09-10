from LLM_client import ConversationSession  # or wherever you land ConversationSession

def run_cli(session_id: str | None = None):
    try:
        print("=" * 60)
        print("FITNESS & WELLNESS RAG CHATBOT")
        print("=" * 60)
        print("Type your questions below. Type 'exit' or 'quit' to end.\n")

        if session_id:
            session = ConversationSession(session_id)
        else:
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
    session_id ="9106e467-3e47-4478-a866-6dbd9b43e7fb"
    # session_id = "a7a"
    run_cli(session_id)
    # run_cli(1)  # Example with a specific session_id, replace with actual ID if needed