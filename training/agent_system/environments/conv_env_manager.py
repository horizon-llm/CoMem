# unify the environment manager for multiple environments
# save the memory in the format of conversation history
from typing import List, Tuple, Dict, Union, Any, Set
from collections import defaultdict
import re
import torch
import numpy as np
from functools import partial
import os
from pathlib import Path
import logging
import json
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import ConversationMemory
from datetime import datetime
from copy import deepcopy

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos

class AlfworldConversationEnvironmentManager(EnvironmentManagerBase):
    """
    A conversational environment manager for Alfworld.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = ConversationMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        self.memory.store({"assistant": [""] * len(text_obs), "user": full_text_obs})  # Initialize with empty assistant responses
        memory_contexts = self.memory.fetch(100000)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos, memory_contexts

    def step(self, text_actions: List[str]):
        """
        The differences between previous manager and this one is that,
        we include the memory contexts as one of our returns.
        The memory contexts are the conversation history made by the agent.
        You may include the memory contexts in the prompt for the next step.
        """
        orig_text_actions = deepcopy(text_actions)
        # Process the actions using the projection function
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        self.memory.store({"assistant": orig_text_actions, "user": full_text_obs})
        memory_contexts = self.memory.fetch(100000)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
        
        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos, memory_contexts

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = f"""The environment returns the following observation: {text_obs[i]}
                Your admissible actions are: {reformatted_admissible_actions}

Now it's your turn to take an action.
You should first reason step-by-step about previous interactions (observations, thinking and actions) and current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""
                # =============== old prompt ===============
#                 obs = f"""The environment returns the following observation: {text_obs[i]}
#                 Your admissible actions are: {reformatted_admissible_actions}

# Now it's your turn to take an action.
# You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
# Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

            postprocess_text_obs.append(obs)
        return postprocess_text_obs
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break
    
    def extract_action(self, text_action: str) -> str:
        """
        Extract the action from the text action.
        The action should be enclosed within <action> </action> tags.
        
        Parameters:
        - text_action (str): The text action from the agent.
        
        Returns:
        - action (str): The extracted action.
        """
        start_tag = "<action>"
        end_tag = "</action>"
        
        start_idx = text_action.find(start_tag)
        end_idx = text_action.find(end_tag)
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return text_action.strip()  # Return the whole action if tags are not found or malformed
        
        action = text_action[start_idx + len(start_tag):end_idx].strip()
        
        return action

    def equal_action(self, action1, action2):
        """
        if search bar, then measure the vocabulary overlap
        else, check the exact match
        """
        a1 = self.extract_action(action1)
        a2 = self.extract_action(action2)
        if a1.startswith("search[") and a2.startswith("search["):
            query1 = a1[len("search["):-1]
            query2 = a2[len("search["):-1]
            overlap_words, overlap_ratio = vocabulary_overlap(query1, query2)
            return overlap_ratio > 0.5
        else:
            return a1 == a2

def vocabulary_overlap(query1: str, query2: str) -> Tuple[Set[str], float]:
    """
    Compute the vocabulary overlap between two search strings.

    Args:
        query1 (str): First search string.
        query2 (str): Second search string.

    Returns:
        overlap_words (set): Words common to both queries.
        overlap_ratio (float): Proportion of overlap relative to the smaller vocabulary size.
    """

    # Basic tokenizer: split on non-alphanumeric characters, lowercase everything
    tokenize = lambda s: set(re.findall(r"\b\w+\b", s.lower()))

    vocab1 = tokenize(query1)
    vocab2 = tokenize(query2)

    # Overlap set
    overlap_words = vocab1.intersection(vocab2)

    # Ratio: overlap relative to the smaller vocabulary
    overlap_ratio = len(overlap_words) / max(1, min(len(vocab1), len(vocab2)))

    return overlap_words, overlap_ratio

class WebshopConversationEnvironmentManager(EnvironmentManagerBase):
    """
    A conversational environment manager for a webshop.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = ConversationMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        text_obs = self.build_text_obs(obs, infos, init=True)
        observations = {'text': text_obs, 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        self.memory.store({"assistant": [""] * len(obs), "user": text_obs})
        memory_contexts = self.memory.fetch(100000)
        return observations, infos, memory_contexts
    
    def step(self, text_actions: List[str]):
        orig_text_actions = deepcopy(text_actions)
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        # self.memory.store({"assistant": orig_text_actions, "user": next_obs})
        self.pre_text_obs = next_obs

        text_obs = self.build_text_obs(next_obs, infos)
        self.memory.store({"assistant": orig_text_actions, "user": text_obs})
        memory_contexts = self.memory.fetch(100000)
        next_observations = {'text': text_obs, 
                            'image': None, 
                            'anchor': next_obs.copy()
                            }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
        
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos, memory_contexts
    
    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks

    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs

    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
    
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = f"""The environment returns the following observation: {text_obs[i]}
                Your admissible actions are: {reformatted_available_actions}

Now it's your turn to take an action.
You should first reason step-by-step about previous interactions (observations, thinking and actions) and current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""
                # =============== old prompt ===============
#                 obs = f"""The environment returns the following observation: {text_obs[i]}
#                 Your admissible actions are: {reformatted_admissible_actions}

# Now it's your turn to take an action.
# You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
# Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return
    
    def extract_action(self, text_action):
        """
        Extract the action from the text action.
        The action should be enclosed within <action> </action> tags.
        
        Parameters:
        - text_action (str): The text action from the agent.
        
        Returns:
        - action (str): The extracted action.
        """
        start_tag = "<action>"
        end_tag = "</action>"
        
        start_idx = text_action.find(start_tag)
        end_idx = text_action.find(end_tag)
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return text_action.strip()

        return text_action[start_idx + len(start_tag):end_idx].strip()

    def equal_action(self, action1, action2):
        """
        if search bar, then measure the vocabulary overlap
        else, check the exact match
        """
        a1 = self.extract_action(action1)
        a2 = self.extract_action(action2)
        if a1.startswith("search[") and a2.startswith("search["):
            query1 = a1[len("search["):-1]
            query2 = a2[len("search["):-1]
            overlap_words, overlap_ratio = vocabulary_overlap(query1, query2)
            return overlap_ratio > 0.5
        else:
            return a1 == a2

class AppWorldConversationEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = ConversationMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, infos = self.envs.reset()

        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        self.memory.store({"assistant": [""] * len(text_obs), "user": full_text_obs})  # Initialize with empty assistant responses
        memory_contexts = self.memory.fetch(100000)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos, memory_contexts

    def step(self, text_actions: List[str]):
        """
        The differences between previous manager and this one is that,
        we include the memory contexts as one of our returns.
        The memory contexts are the conversation history made by the agent.
        You may include the memory contexts in the prompt for the next step.
        """
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)
        self.memory.store({"assistant": text_actions, "user": full_text_obs})
        memory_contexts = self.memory.fetch(100000)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
        
        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos, memory_contexts
    
    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                obs = f"""The environment returns the following observation: {text_obs[i]}

Now it's your turn to take an action.
You should first reason step-by-step about previous interactions (observations, thinking and actions) and current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reflexion and reasoning, you present the solution code body within <code> </code> tags.
"""
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

class SearchConversationEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = ConversationMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs: List[Dict]) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        text_obs = self.build_text_obs(obs, init=True)

        self.memory.reset(batch_size=len(obs))

        self.memory.store({"assistant": [""] * len(obs), "user": text_obs})
        memory_contexts = self.memory.fetch(100000)

        observations = {
            "text": text_obs,
            "image": None,
            "anchor": obs.copy()
        }

        return observations, infos, memory_contexts

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        text_obs = self.build_text_obs(next_obs)
        self.memory.store({
            "assistant": actions,
            "user": text_obs,
        })
        memory_contexts = self.memory.fetch(100000)

        next_observations = {
            "text": text_obs,
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos, memory_contexts
    
    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        for i in range(len(text_obs)):
            if init:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
                obs_i = f"""You are an expert agent tasked with answering the given question step-by-step.
Your question: {self.tasks[i]}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) Generate search queries for external information. For example, <search>what's the capital of china</search>.
(2) Answer the question with a simple entity. For example, <answer>Beijing</answer>.
Any other actions are not allowed and will return no observations."""
            else:
                if text_obs[i].strip() == "":
                    actual_obs = "Invalid action."
                else:
                    actual_obs = text_obs[i]
#                 obs_i = f"""The environment returns the following observation: {actual_obs}

# Now it's your turn to respond for the current step.
# You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
# After completing your reasoning, choose only one of the following actions (do not perform both):
# (1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
# (2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>."""
                obs_i = f"""The environment returns the following observation: {actual_obs}
Your question: {self.tasks[i]}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) Generate search queries for external information. For example, <search>what's the capital of china</search>.
(2) Answer the question with a simple entity. For example, <answer>Beijing</answer>.
Any other actions are not allowed and will return no observations."""
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask

class MultiSearchConversationEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = ConversationMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs: List[Dict]) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        text_obs = self.build_text_obs(obs, init=True)

        self.memory.reset(batch_size=len(obs))

        self.memory.store({"assistant": [""] * len(obs), "user": text_obs})
        memory_contexts = self.memory.fetch(100000)

        observations = {
            "text": text_obs,
            "image": None,
            "anchor": obs.copy()
        }

        return observations, infos, memory_contexts

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        text_obs = self.build_text_obs(next_obs)
        self.memory.store({
            "assistant": actions,
            "user": text_obs,
        })
        memory_contexts = self.memory.fetch(100000)

        next_observations = {
            "text": text_obs,
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos, memory_contexts
    
    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        for i in range(len(text_obs)):
            if init:
                obs_i = f"""You are an expert agent tasked with answering the given questions step-by-step.
Your questions: {self.tasks[i]}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
(2) If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.
Any other actions are not allowed and will return no observations."""
            else:
                if text_obs[i].strip() == "":
                    actual_obs = "Invalid action."
                else:
                    actual_obs = text_obs[i]
#                 obs_i = f"""The environment returns the following observation: {actual_obs}

# Now it's your turn to respond for the current step.
# You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
# After completing your reasoning, choose only one of the following actions (do not perform both):
# (1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
# (2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>."""
                obs_i = f"""The environment returns the following observation: {actual_obs}
Your question: {self.tasks[i]}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
(2) If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.
Any other actions are not allowed and will return no observations."""
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask

    def extract_action(self, text_action: str) -> str:
        """
        Extract the action from the text action.
        The action can be enclosed within <search> </search> or <answer> </answer> tags.

        Parameters:
        - text_action (str): The text action from the agent.

        Returns:
        - action (str): The extracted action with the appropriate tags.
        """
        # Try to extract <search> action first
        search_start_tag = "<search>"
        search_end_tag = "</search>"
        search_start_idx = text_action.find(search_start_tag)
        search_end_idx = text_action.find(search_end_tag)

        if search_start_idx != -1 and search_end_idx != -1 and search_start_idx < search_end_idx:
            search_content = text_action[search_start_idx + len(search_start_tag):search_end_idx].strip()
            return f"{search_start_tag}{search_content}{search_end_tag}"

        # Try to extract <answer> action
        answer_start_tag = "<answer>"
        answer_end_tag = "</answer>"
        answer_start_idx = text_action.find(answer_start_tag)
        answer_end_idx = text_action.find(answer_end_tag)

        if answer_start_idx != -1 and answer_end_idx != -1 and answer_start_idx < answer_end_idx:
            answer_content = text_action[answer_start_idx + len(answer_start_tag):answer_end_idx].strip()
            return f"{answer_start_tag}{answer_content}{answer_end_tag}"

        # If no valid tags found, return the whole action stripped
        return text_action.strip()

    def equal_action(self, action1: str, action2: str) -> bool:
        """
        Compare two actions for equality.
        For search actions, measure vocabulary overlap.
        For answer actions, check exact match.

        Parameters:
        - action1 (str): First action.
        - action2 (str): Second action.

        Returns:
        - bool: True if actions are considered equal, False otherwise.
        """
        a1 = self.extract_action(action1)
        a2 = self.extract_action(action2)

        # Both are search actions - use vocabulary overlap
        if a1.startswith("<search>") and a2.startswith("<search>"):
            query1 = a1[len("<search>"):-len("</search>")]
            query2 = a2[len("<search>"):-len("</search>")]
            overlap_words, overlap_ratio = vocabulary_overlap(query1, query2)
            return overlap_ratio > 0.5

        # Both are answer actions or other types - use exact match
        return a1 == a2

class SciworldConversationEnvironmentManager(EnvironmentManagerBase):
    """
    A conversational environment manager for Sciworld.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = ConversationMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        obs, infos = self.envs.reset()
        self.tasks = []
        self.extract_task(obs)
        text_obs = self.build_text_obs(obs, init=True)
        observations = {'text': text_obs, 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        self.memory.store({"assistant": [""] * len(obs), "user": text_obs})
        memory_contexts = self.memory.fetch(100000)
        return observations, infos, memory_contexts
    
    def step(self, text_actions: List[str]):
        orig_text_actions = deepcopy(text_actions)
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # self.memory.store({"assistant": orig_text_actions, "user": next_obs})
        self.pre_text_obs = next_obs

        text_obs = self.build_text_obs(next_obs, init=False)
        self.memory.store({"assistant": orig_text_actions, "user": text_obs})
        memory_contexts = self.memory.fetch(100000)
        next_observations = {'text': text_obs, 
                            'image': None, 
                            'anchor': next_obs.copy()
                            }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
        
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos, memory_contexts
    
    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
            
        for i in range(len(text_obs)):
            
            if init:
                obs = SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                )
            else:
                obs = f"""The environment returns the following observation: {text_obs[i]}
                Your admissible actions are:
                [
{{"action": "open OBJ", "description": "open a container"}}, {{"action": "close OBJ", "description": "close a container"}}, {{"action": "activate OBJ", "description": "activate a device"}}, {{"action": "deactivate OBJ", "description": "deactivate a device"}}, {{"action": "connect OBJ to OBJ", "description": "connect electrical components"}}, {{"action": "disconnect OBJ", "description": "disconnect electrical components"}}, {{"action": "use OBJ [on OBJ]", "description": "use a device/item"}}, {{"action": "look around", "description": "describe the current room"}}, {{"action": "look at OBJ", "description": "describe an object in detail"}}, {{"action": "look in OBJ", "description": "describe a container\'s contents"}}, {{"action": "read OBJ", "description": "read a note or book"}}, {{"action": "move OBJ to OBJ", "description": "move an object to a container"}}, {{"action": "pick up OBJ", "description": "move an object to the inventory"}}, {{"action": "put down OBJ", "description": "drop an inventory item"}}, {{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}}, {{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}}, {{"action": "mix OBJ", "description": "chemically mix a container"}}, {{"action": "go to LOC", "description": "move to a new location"}}, {{"action": "eat OBJ", "description": "eat a food"}}, {{"action": "flush OBJ", "description": "flush a toilet"}}, {{"action": "focus on OBJ", "description": "signal intent on a task object"}}, {{"action": "wait", "description": "take no action for 10 iterations"}}, {{"action": "wait1", "description": "take no action for 1 iteration"}}, {{"action":"examine OBJ","description":"provides a description of the objects present on or in a receptacle."}}, {{"action": "task", "description": "describe current task"}}, {{"action": "inventory", "description": "list your inventory"}}
].

Now it's your turn to take an action.
You should first reason step-by-step about previous interactions (observations, thinking and actions) and current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""
            postprocess_text_obs.append(obs)

        return postprocess_text_obs
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to ')
            task_end = obs.find('.')

            if task_start != -1 and task_end != -1:
                self.tasks.append(obs[task_start + len('Your task is to '):task_end].strip())
            else:
                raise ValueError("Task description not found in text observation.")
    
    def extract_action(self, text_action):
        """
        Extract the action from the text action.
        The action should be enclosed within <action> </action> tags.
        
        Parameters:
        - text_action (str): The text action from the agent.
        
        Returns:
        - action (str): The extracted action.
        """
        start_tag = "<action>"
        end_tag = "</action>"
        
        start_idx = text_action.find(start_tag)
        end_idx = text_action.find(end_tag)
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return text_action.strip()

        return text_action[start_idx + len(start_tag):end_idx].strip()

    def equal_action(self, action1, action2):
        """
        if search bar, then measure the vocabulary overlap
        else, check the exact match
        """
        a1 = self.extract_action(action1)
        a2 = self.extract_action(action2)
        if a1.startswith("search[") and a2.startswith("search["):
            query1 = a1[len("search["):-1]
            query2 = a2[len("search["):-1]
            overlap_words, overlap_ratio = vocabulary_overlap(query1, query2)
            return overlap_ratio > 0.5
        else:
            return a1 == a2
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['sciworld_task_score (not success_rate)'].append(score_value)
                return

class BabyAIConversationEnvironmentManager(EnvironmentManagerBase):
    """
    A conversational environment manager for BabyAI.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = ConversationMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        obs, infos = self.envs.reset()
        self.tasks = []
        self.extract_task(obs)
        text_obs = self.build_text_obs(obs, init=True)
        observations = {'text': text_obs, 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        self.memory.store({"assistant": [""] * len(obs), "user": text_obs})
        memory_contexts = self.memory.fetch(100000)
        return observations, infos, memory_contexts

    def step(self, text_actions: List[str]):
        orig_text_actions = deepcopy(text_actions)
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # self.memory.store({"assistant": orig_text_actions, "user": next_obs})
        self.pre_text_obs = next_obs

        text_obs = self.build_text_obs(next_obs, infos)
        self.memory.store({"assistant": orig_text_actions, "user": text_obs})
        memory_contexts = self.memory.fetch(100000)
        next_observations = {'text': text_obs, 
                            'image': None, 
                            'anchor': next_obs.copy()
                            }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
        
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos, memory_contexts

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
            
        for i in range(len(text_obs)):
            
            if init:
                obs = BABYAI_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                )
            else:
                obs = f"""The environment returns the following observation: {text_obs[i]}
                Your admissible actions are:
                - turn right 
- turn left 
- move forward
- go to <obj> <id> 
- pick up <obj> <id> 
- go through <door> <id>: <door> must be an open door. 
- toggle and go through <door> <id>: <door> can be a closed door or a locked door. If you want to open a locked door, you need to carry a key that is of the same color as the locked door. 
- toggle: there is a closed or locked door right in front of you and you can toggle it.

Now it's your turn to take an action.
You should first reason step-by-step about previous interactions (observations, thinking and actions) and current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
            postprocess_text_obs.append(obs)

        return postprocess_text_obs
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your goal: ')
            task_end = obs.find('\n')

            if task_start != -1 and task_end != -1:
                self.tasks.append(obs[task_start + len('Your goal: '):task_end].strip())
            else:
                raise ValueError("Task description not found in text observation.")

class TextCraftConversationEnvironmentManager(EnvironmentManagerBase):
    """
    A conversational environment manager for TextCraft.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = ConversationMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        obs, infos = self.envs.reset()
        self.tasks = []
        self.extract_task(obs)
        text_obs = self.build_text_obs(obs, init=True)
        observations = {'text': text_obs, 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        self.memory.store({"assistant": [""] * len(obs), "user": text_obs})
        memory_contexts = self.memory.fetch(100000)
        return observations, infos, memory_contexts

    def step(self, text_actions: List[str]):
        orig_text_actions = deepcopy(text_actions)
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # self.memory.store({"assistant": orig_text_actions, "user": next_obs})
        self.pre_text_obs = next_obs

        text_obs = self.build_text_obs(next_obs, infos)
        self.memory.store({"assistant": orig_text_actions, "user": text_obs})
        memory_contexts = self.memory.fetch(100000)
        next_observations = {'text': text_obs, 
                            'image': None, 
                            'anchor': next_obs.copy()
                            }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
        
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos, memory_contexts

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
            
        for i in range(len(text_obs)):
            
            if init:
                obs = TEXTCRAFT_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                )
            else:
                obs = f"""The environment returns the following observation: {text_obs[i]}

Now it's your turn to take an action.
You should first reason step-by-step about previous interactions (observations, thinking and actions) and current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
            postprocess_text_obs.append(obs)
        
        return postprocess_text_obs

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Goal: ')

            if task_start != -1:
                self.tasks.append(obs[task_start + len('Goal: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

def make_conv_envs(config):
    """
    Create environments
    """
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "gym_cards" in config.env.env_name.lower():
        raise NotImplementedError("Gym cards environment is not supported yet.")
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")
        
        env_kwargs = {
            'eval_dataset': 'eval_in_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(alfworld_projection)
        envs = AlfworldConversationEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfworldConversationEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        raise NotImplementedError("Sokoban environment is not supported yet.")
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.subset == "small":
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        elif config.env.webshop.subset == "10k":
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/subsets/10k/items_shuffle_10000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/subsets/10k/items_ins_v2_10000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection)
        envs = WebshopConversationEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopConversationEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldConversationEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldConversationEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchConversationEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchConversationEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "multi_search" in config.env.env_name.lower():
        from agent_system.environments.env_package.multi_search import build_multi_search_envs, multi_search_projection
        _envs = build_multi_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_multi_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(multi_search_projection)
        envs = MultiSearchConversationEnvironmentManager(_envs, projection_f, config)
        val_envs = MultiSearchConversationEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sciworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
        _envs = build_sciworld_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs={"env_server_base": config.env.env_server_base})
        _val_envs = build_sciworld_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs={"env_server_base": config.env.env_server_base})

        projection_f = partial(sciworld_projection)
        envs = SciworldConversationEnvironmentManager(_envs, projection_f, config)
        val_envs = SciworldConversationEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "babyai" in config.env.env_name.lower():
        from agent_system.environments.env_package.babyai import build_babyai_envs, babyai_projection
        _envs = build_babyai_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs={"env_server_base": config.env.env_server_base})
        _val_envs = build_babyai_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs={"env_server_base": config.env.env_server_base})

        projection_f = partial(babyai_projection)
        envs = BabyAIConversationEnvironmentManager(_envs, projection_f, config)
        val_envs = BabyAIConversationEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "textcraft" in config.env.env_name.lower():
        from agent_system.environments.env_package.textcraft import build_textcraft_envs, textcraft_projection
        _envs = build_textcraft_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs={"env_server_base": config.env.env_server_base})
        _val_envs = build_textcraft_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs={"env_server_base": config.env.env_server_base})

        projection_f = partial(textcraft_projection)
        envs = TextCraftConversationEnvironmentManager(_envs, projection_f, config)
        val_envs = TextCraftConversationEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)