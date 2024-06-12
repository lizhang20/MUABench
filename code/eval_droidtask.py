import os
import re
from typing import List, NamedTuple

import yaml
from openai import OpenAI


class Task(NamedTuple):
    task_description: str
    ui_representation: str
    action_index: int
    total_index: int


def load_tasks_from_file(filepath: str):
    """
    Given a single yaml file, load a list of tasks.
    Every individual task contain the following fields:
        1. task description: str
        2. UI representation in html: str
        3. groundtruth results: choice: int and param: str
    """

    tasks: List[Task] = []
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

        task_description = data["task_name"]

        for record in data["records"]:
            if record["Input"] != "null":
                continue
            assert (
                record["Input"] == "null"
            ), f"Record text input is not null: {record['Input']}"

            line_last = record["State"].splitlines()[-1]
            total_index = re.search(r"id=(\d+)", line_last).group(1)
            total_index = int(total_index) + 1

            action_index = int(record["Choice"])
            action_index = action_index if action_index != -1 else total_index
            # print(f"UI = {record['State']}\nTotal = {total_index}")

            tasks.append(
                Task(
                    task_description=task_description,
                    ui_representation=record["State"],
                    action_index=action_index,
                    total_index=total_index,
                )
            )

    return tasks


def load_all_tasks(task_path: str):
    apps = [
        "applauncher",
        "calendar",
        "camera",
        "clock",
        "contacts",
        "dialer",
        "filemanager",
        "firefox",
        "gallery",
        "messenger",
        "musicplayer",
        "notes",
        "voicerecorder",
    ]

    task_files = []
    app_paths = [os.path.join(task_path, app) for app in apps]
    for app_path in app_paths:
        assert os.path.exists(app_path)
        for file in os.listdir(app_path):
            if file.endswith(".yaml") and "task" in file:
                task_files.append(os.path.join(app_path, file))

    print(f"{len(task_files)} task files detected")

    all_tasks = []
    for tf in task_files:
        all_tasks += load_tasks_from_file(tf)

    print(f"{len(all_tasks)} tasks loaded")

    return all_tasks


def construct_query_prompt_zero_shot(task: Task):
    prompt = f"""\
The simplified UI can be represented as: ```{task.ui_representation}```.
I want to execute the task: ```{task.task_description}```.
I need to tap the button with ID: 
"""
    return prompt


def extract_and_compare_query_results(response: str, task: Task) -> bool:
    try:
        id = int(response)
    except ValueError:
        return False
    return id == task.action_index


def query_llm(prompt: str) -> str:
    """Query the LLM which has been tuned by instructions."""

    SYSPROMPT = "Only output a number.\n"
    prompt = SYSPROMPT + prompt
    # return "1"

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        seed=0,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt},
        ],
        timeout=30,
    )
    response = completion.choices[0].message.content
    return response


def test_with_local_model():
    """mmlu
    doc_to_text: construct_query_prompt_zero_shot(task)
    doc_to_choice: [for i in range(0, task.total_index)]
    doc_to_target: task.action_index
    """
    tasks: List[Task] = load_all_tasks("DroidTask")

    for index, task in enumerate(tasks):
        prompt = construct_query_prompt_zero_shot(task)
        response = query_llm(prompt)
        print(
            f"task-{index}, success: {extract_and_compare_query_results(response, task)}"
        )


if __name__ == "__main__":
    test_with_local_model()
