from graph_flow import build_graph

def run_chatbot():
    print("======================================")
    print(" Customer Support Assistant (RAG)")
    print(" Type 'exit' to quit")
    print("======================================\n")

    app = build_graph()

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Thank you!")
            break

        app.invoke({
            "query": query,
            "answer": ""
        })


if __name__ == "__main__":
    run_chatbot()