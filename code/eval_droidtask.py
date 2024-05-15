import os
import re
from typing import List, NamedTuple

import ollama
import yaml
from openai import OpenAI


class Task(NamedTuple):
    task_description: str
    ui_representation: str
    action_index: int
    action_param: str


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
            tasks.append(
                Task(
                    task_description=task_description,
                    ui_representation=record["State"],
                    action_index=record["Choice"],
                    action_param=record["Input"],
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
    prompt = f"""Given a screen, an instruction, predict the id of the UI element to perform or continue the instruction.
If you think it needs to input text, also give the text, else return the string 'null'.
Your output should only contain the folliwing information:
[id], '[input]'

Screen:
{task.ui_representation}

Instruction: {task.task_description}
"""
    return prompt


def construct_query_prompt_one_shot(task: Task):
    prompt = f"""Given a screen, an instruction, predict the id of the UI element to perform or continue the instruction.
If you think it needs to input text, also give the text, else give the string 'null'.
Your output should only contain the folliwing information:
[id], '[input]'

Screen:
{task.ui_representation}

Instruction: {task.task_description}

The following are examples:

Screen:
<input id=0 >Search</input>\n <button id=1 text='Sort by'></button>\n <button\
    \ id=2 text='Toggle app name visibility'></button>\n <button id=3 text='More options'></button>\n\
    \ <button id=4>Calendar</button>\n <button id=5>Camera</button>\n <button id=6>Clock</button>\n\
    \ <button id=7>go back</button>

Instruction: Sort apps by title in descending order

Output:
1, 'null'
"""
    return prompt


def construct_query_prompt_two_shots(task: Task):
    prompt = f"""Given a screen, an instruction, predict the id of the UI element to perform or continue the instruction.
If you think it needs to input text, also give the text, else give the string 'null'.
Your output should only contain the folliwing information:
[id], '[input]'

Screen:
{task.ui_representation}

Instruction: {task.task_description}

The following are examples:

Example 1:

Screen:
<input id=0 >Search</input>\n <button id=1 text='Sort by'></button>\n <button\
    \ id=2 text='Toggle app name visibility'></button>\n <button id=3 text='More options'></button>\n\
    \ <button id=4>Calendar</button>\n <button id=5>Camera</button>\n <button id=6>Clock</button>\n\
    \ <button id=7>go back</button>

Instruction: Sort apps by title in descending order

Output:
1, 'null'

Example 2:

Screen:
<button id=0 text=''Search''></button>

    <button id=1 text=''Open note''></button>

    <button id=2 text=''Create a new note''></button>

    <button id=3 text=''More options''></button>

    <button id=4>General note</button>

    <p id=5>test</p>

    <input id=6>Insert text here</input>

    <button id=7>go back</button>

Instruction: Create a text note called 'test', type '12345678', and search for '234'

Output:
6, '123456'
"""
    return prompt


def extract_and_compare_query_results(response: str, task: Task) -> bool:
    try:
        id, input = int(response.split(", ")[0]), response.split(", ")[1].strip(
            "'"
        ).strip('"')
    except ValueError:
        return False

    # print(f"extracting {id} {input} from response: {response}")
    if id == task.action_index and input == task.action_param:
        return True

    return False


def test_with_local_model():
    tasks: List[Task] = load_all_tasks("../DroidTask")

    for index, task in enumerate(tasks[:10]):
        prompt = construct_query_prompt_one_shot(task)
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

        print(
            f"task-{index}, success: {extract_and_compare_query_results(response, task)}"
        )


if __name__ == "__main__":
    test_with_local_model()
