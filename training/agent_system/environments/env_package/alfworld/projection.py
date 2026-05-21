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

from typing import List
import re

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within action tags (flexible format)
        # Try multiple patterns to handle format variations from LLMs
        action_patterns = [
            (r'<action>(.*?)</action>', 8, 9),  # <action>content</action>
            (r'\[action\](.*?)\[/action\]', 8, 9),  # [action]content[/action]
            (r'%action>(.*?)<%/action>', 8, 10),  # %action>content<%/action>
            (r'\$action>(.*?)<\$/action>', 8, 11),  # $action>content<$/action>
            (r'<action>(.*?)<%/action>', 8, 10),  # <action>content<%/action>
            (r'\[action\](.*?)</action>', 8, 9),  # [action]content</action>
            (r'action:(.*?)(?:\n|$)', 7, 0),  # action: content (line ending)
            (r'action\s*=\s*(.*?)(?:\n|$)', 0, 0),  # action = content (line ending)
        ]
        
        extracted_action = None
        try:
            for pattern, _, _ in action_patterns:
                match = re.search(pattern, actions[i], re.IGNORECASE | re.DOTALL)
                if match:
                    extracted_action = match.group(1).strip().lower()
                    break
            
            if extracted_action is None:
                # If we can't find any valid action pattern, mark as invalid
                actions[i] = actions[i][-30:]  # invalid action
                continue

            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check <think>...</think> with flexible format
        think_patterns = [
            r'<think>.*?</think>',
            r'\[think\].*?\[/think\]', 
            r'%think>.*?<%/think>',
            r'\$think>.*?<\$/think>',
            r'<think>.*?<%/think>',
            r'\[think\].*?</think>',
            r'think:.*?(?:\n|$)',
        ]
        
        has_think = False
        for pattern in think_patterns:
            if re.search(pattern, original_str, re.IGNORECASE | re.DOTALL):
                has_think = True
                break
        
        if not has_think:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids
