def handle_human_escalation(query):
    print("\n[HITL] Escalation triggered")
    print("Query:", query)

    response = input("Enter human response: ")

    return response