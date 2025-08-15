from deep_researcher.graph import agent_graph
from IPython.display import Image, display
import uuid
import os
import pandas as pd

litqa2 = pd.read_csv("evals/litqa2.tsv", sep="\t")
litqa2 = litqa2[litqa2['is_opensource'] == True]
questions = litqa2["question"][0:3].to_list()

thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "max_queries": 3,
            "search_depth": 3,
            "num_reflections": 3,
            "temperature": 1
        }
    }

os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

for question in questions:
    for event in agent_graph.stream(
            {"topic": "Molecular Biology", "outline": question},
            config=thread,
    ):
        with open("logs/agent_logs.txt", "a", encoding="utf-8") as f:
            f.write(str(event))
            f.write("\n\n\n")

        if "report_structure_planner" in event:
            print("<<< REPORT STRUCTURE PLANNER >>>")
            print(event["report_structure_planner"]["messages"][-1].content)
            print("\n", "=" * 100, "\n")
        elif "section_formatter" in event:
            pass
        elif "research_agent" in event:
            pass
        elif "human_feedback" in event:
            print("<<< HUMAN FEEDBACK >>>")
            print(event["human_feedback"]["messages"][-1].content)
            print("\n", "=" * 100, "\n")
        elif "queue_next_section" in event:
            pass
        elif "finalizer" in event:
            print("Final report complete.")
        else:
            print("<<< UNKNOWN EVENT >>>")
            print(event)
            print("\n", "=" * 100, "\n")



