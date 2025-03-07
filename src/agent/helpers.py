from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

def agent_node(state, agent, name):
    # Debugging: Print the messages before invoking the agent
    print("Messages before API call:", state["messages"])

    # Ensure the last message is from the user
    if not isinstance(state["messages"][-1], HumanMessage):
        state["messages"].append(HumanMessage(content="Please proceed with the task."))

    # Invoke the agent
    result = agent.invoke(state)

    # Debugging: Print the agent's output
    print("Agent output:", result)

    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    return {
        "messages": [result],
        "sender": name,
    }