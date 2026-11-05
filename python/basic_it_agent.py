'''
This is basic implementation if an IT agent who accepts user input and invoke configured tools to solve issue
'''

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_APIKEY'),
)

def get_server_health(server_id: str) -> str:
    """Returns CPU and Memory usage for a given server."""
    print(f"-> TOOL: Checking health for {server_id}...")

    metrics = {
        # Scenario 1: High CPU (Needs Restart)
        "payment-server-01": {"cpu": "98%", "memory": "40%", "status": "Warning"},

        # Scenario 2: Healthy (No Action Needed)
        "db-node-02": {"cpu": "12%", "memory": "60%", "status": "Healthy"},

        # Scenario 3: High Memory Leak (Needs Restart or Escalation)
        "auth-service-03": {"cpu": "45%", "memory": "95%", "status": "Critical"},

        # Scenario 4: Network/Dependency Failure (Needs Escalation)
        "search-index-09": {"cpu": "10%", "memory": "15%", "status": "Error"},

        # Scenario 5: Completely Normal
        "frontend-node-04": {"cpu": "25%", "memory": "30%", "status": "Healthy"},
    }

    result = metrics.get(server_id, {"error": "Server not found. Check the ID."})
    return json.dumps(result)

def fetch_recent_logs(server_id: str, lines: int = 5) -> str:
    """Returns the last N lines of logs."""
    print(f"-> TOOL: Fetching last {lines} log lines for {server_id}...")

    # Different logs for different servers to trigger different agent behaviors
    log_database = {
        "payment-server-01": [
            "[INFO] Request received /pay/v1",
            "[WARN] CPU threshold exceeded 90%",
            "[WARN] Thread pool exhaustion",
            "[CRITICAL] Process hung, not accepting new connections",
            "[ERROR] Timeout waiting for thread"
        ],
        "db-node-02": [
            "[INFO] Backup started",
            "[INFO] Backup completed successfully",
            "[INFO] User query executed in 12ms",
            "[INFO] Health check: OK",
            "[INFO] Replication sync active"
        ],
        "auth-service-03": [
            "[INFO] Token validated user_882",
            "[WARN] Garbage collection taking too long (>5s)",
            "[ERROR] java.lang.OutOfMemoryError: Java heap space",
            "[CRITICAL] Application crashing due to memory leak",
            "[INFO] Restarting context..."
        ],
        "search-index-09": [
            "[INFO] Indexing started",
            "[ERROR] Connection refused: elastic-cluster-main:9200",
            "[ERROR] Failed to write document ID 4432",
            "[CRITICAL] Dependency Unreachable: Search Engine is down",
            "[ERROR] Retrying in 30s..."
        ],
        "frontend-node-04": [
            "[INFO] GET /home 200 OK",
            "[INFO] GET /assets/logo.png 200 OK",
            "[INFO] GET /login 200 OK",
            "[INFO] GET /api/v1/status 200 OK",
            "[INFO] Health check passed"
        ]
    }

    # Default logs if server not found in specific list
    default_logs = ["[INFO] System stable", "[INFO] Heartbeat signal received"]

    logs = log_database.get(server_id, default_logs)
    return json.dumps({"logs": logs[:lines]})

def restart_service(server_id: str) -> str:
    """Restarts service"""
    print("-> TOOL: Restarting service...")
    return json.dumps({"status": "success", "message": "Server restarted successfully"})


def escalate_to_engineer(summary: str) -> str:
    """ Excalate to engineer"""
    print("-> TOOL: Escalating to human...")
    return json.dumps({"status": "success", "message": "Ticket was created"})

AVAILABLE_FUNCTIONS = {
    "get_server_health": get_server_health,
    "fetch_recent_logs": fetch_recent_logs,
    "restart_service": restart_service,
    "escalate_to_engineer": escalate_to_engineer,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_server_health",
            "description": "Checks the current CPU and memory usage of a specific server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "The ID of the server, e.g., 'payment-server-01'"}
                },
                "required": ["server_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_recent_logs",
            "description": "Retrieves the most recent log entries from a server to diagnose errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "The ID of the server."},
                    "lines": {"type": "integer", "description": "Number of log lines to fetch."}
                },
                "required": ["server_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restarts service on given server", 
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "The ID of the server, e.g., 'payment-server-01'"}
                },
                "required": ["server_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_engineer",
            "description": "Escalates the issue to a human engineer when automated fixes fail or the error is unknown.",
            "parameters": {
                "type": "object",
                "properties": {
                     "summary": {"type": "string", "description": "Summary of the issue"}
                },
                "required": ["summary"]
            }
        }
    }
]

SYSTEM_PROMPT = ("You are a Level 1 IT Responder. Investigate server issues. "
                 "If CPU or Memory is > 90%, restart the service. If logs show critical dependency errors (like connection refused) that a restart won't fix, escalate to an engineer.")

MODEL = "openai/gpt-6-luna"   # cheap OpenRouter model; other options: "openai/gpt-5.4-nano", "anthropic/claude-haiku-5.5"

def run_it_agent(user_issue: str, max_steps: int = 10):
    """Runs the agent loop. Returns the final answer and the trace of tool calls (used by evals)."""
    print(f"\n--- New Incident: {user_issue} ---")
    tool_trace = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_issue}
    ]

    for _ in range(max_steps):
        print("\n[AI Thinking...]")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )

        response_msg = response.choices[0].message
        messages.append(response_msg)

        if response_msg.tool_calls:
            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_trace.append({"name": func_name, "args": func_args})

                # Retrieve the actual python function based on name
                function_to_call = AVAILABLE_FUNCTIONS.get(func_name)

                if function_to_call:
                    tool_output = function_to_call(**func_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id, 
                        "name": func_name,         
                        "content": tool_output       
                    })

        else:
            print(f"\n[FINAL RESPONSE]: {response_msg.content}")
            return {"final": response_msg.content, "tool_calls": tool_trace, "hit_max_steps": False}

    return {"final": None, "tool_calls": tool_trace, "hit_max_steps": True}

if __name__ == "__main__":
    while True:
    # Sample user input:
    #     The payment-server-01 is extremely slow and timing out.
    #     Users are reporting login failures on auth-service-03.
    #     Search isn't working. Can you check search-index-09?
    #     Check frontend-node-04 just to be safe.
        user_input = input("\nHow can I help you : ")
        run_it_agent(user_input)
        print("\n" + "="*50 + "\n")

