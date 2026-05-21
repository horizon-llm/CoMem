# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- SciWorld --------------------- #
SCIWORLD_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the science world environment. Your objective is to complete the science-themed task by interacting with a text-based environment. You control an agent using short, imperative commands (one per turn). At each step, the environment returns an observation describing what you see and the results of your action. Use only valid actions and objects available in the current state. Here are the actions you may take: [{{"action": "open OBJ", "description": "open a container"}}, {{"action": "close OBJ", "description": "close a container"}}, {{"action": "activate OBJ", "description": "activate a device"}}, {{"action": "deactivate OBJ", "description": "deactivate a device"}}, {{"action": "connect OBJ to OBJ", "description": "connect electrical components"}}, {{"action": "disconnect OBJ", "description": "disconnect electrical components"}}, {{"action": "use OBJ [on OBJ]", "description": "use a device/item"}}, {{"action": "look around", "description": "describe the current room"}}, {{"action": "look at OBJ", "description": "describe an object in detail"}}, {{"action": "look in OBJ", "description": "describe a container\'s contents"}}, {{"action": "read OBJ", "description": "read a note or book"}}, {{"action": "move OBJ to OBJ", "description": "move an object to a container"}}, {{"action": "pick up OBJ", "description": "move an object to the inventory"}}, {{"action": "put down OBJ", "description": "drop an inventory item"}}, {{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}}, {{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}}, {{"action": "mix OBJ", "description": "chemically mix a container"}}, {{"action": "go to LOC", "description": "move to a new location"}}, {{"action": "eat OBJ", "description": "eat a food"}}, {{"action": "flush OBJ", "description": "flush a toilet"}}, {{"action": "focus on OBJ", "description": "signal intent on a task object"}}, {{"action": "wait", "description": "take no action for 10 iterations"}}, {{"action": "wait1", "description": "take no action for 1 iteration"}}, {{"action":"examine OBJ","description":"provides a description of the objects present on or in a receptacle."}}, {{"action": "task", "description": "describe current task"}}, {{"action": "inventory", "description": "list your inventory"}}]\n
Your task is to: {task_description}.
Your current observation is: {current_observation}.

Now it's your turn to take one action for the current step.
You should first reason step-by-step about object properties (e.g., temperature, state of matter, containment, power), locations, and causal effects to achieve the goal efficiently. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""