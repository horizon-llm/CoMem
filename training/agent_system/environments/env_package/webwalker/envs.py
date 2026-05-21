"""
WebWalker Gym-style Environment
Provides a standard RL interface with step() and reset() methods for web navigation tasks.
"""

import asyncio
import json
import os
from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
import sys

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from agent import WebWalker
from utils import get_info
# from app import extract_links_with_text
# from qwen_agent.llm.schema import Message


class WebWalkerEnv:
    """
    Gym-style environment wrapper for WebWalker agent.
    
    Provides a standardized interface for web navigation with:
    - reset(): Initialize/reset the environment
    - step(action): Take an action and return (observation, reward, done, info)
    """
    
    def __init__(
        self,
        llm_cfg: Dict[str, Any],
        max_steps: int = 10,
        tools: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the WebWalker environment.
        
        Args:
            llm_cfg: LLM configuration dictionary with keys:
                - model: Model name (e.g., 'gpt-4', 'qwen-plus')
                - api_key: API key
                - model_server: Model server URL
                - generate_cfg: Generation configuration
            max_steps: Maximum number of steps per episode
            tools: List of tool names to use (default: ['visit_page'])
            **kwargs: Additional arguments
        """
        self.llm_cfg = llm_cfg
        self.max_steps = max_steps
        self.tools = tools or ["visit_page"]
        self.kwargs = kwargs
        
        # Initialize agent
        self.agent = None
        
        # Environment state
        self.current_step = 0
        self.task = None
        self.root_url = None
        self.current_url = None
        self.done = False
        self.trajectory = []
        self.messages = []
        self.memory = []
        
        # Track visited pages
        self.visited_urls = set()
        self.button_url_mapping = {}
        
    def reset(self, task: str, root_url: str) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.
        
        Args:
            task: The web navigation task/query to accomplish
            root_url: Starting URL for web navigation
            
        Returns:
            observation: Initial observation containing current state
        """
        # Reset state
        self.current_step = 0
        self.done = False
        self.trajectory = []
        self.messages = []
        self.memory = []
        self.task = task
        self.root_url = root_url
        self.current_url = root_url
        self.visited_urls = set()
        self.button_url_mapping = {}
        
        # Update LLM config with task and action count
        self.llm_cfg["query"] = task
        self.llm_cfg["action_count"] = self.max_steps
        
        # Initialize WebWalker agent
        self.agent = WebWalker(
            llm=self.llm_cfg,
            function_list=self.tools
        )
        
        # Save root URL to file (required by WebWalker)
        with open("ROOT_URL.txt", "w") as f:
            f.write(self.root_url)
        
        # Initialize button URL mapping
        if os.path.exists("BUTTON_URL_ADIC.json"):
            os.remove("BUTTON_URL_ADIC.json")
        with open("BUTTON_URL_ADIC.json", "w") as f:
            json.dump({}, f)
        
        # Get initial observation from root URL
        html, markdown, screenshot = asyncio.run(get_info(self.root_url))
        
        # Extract clickable buttons
        buttons = self._extract_buttons(html)
        
        # Build initial observation
        observation_text = f"website information:\n\n{markdown}\n\n"
        observation_text += f"clickable button:\n\n{buttons}\n\nEach button is wrapped in a <button> tag"
        
        # Create initial prompt
        start_prompt = f"query:\n{task}\nofficial website:\n{self.root_url}"
        start_prompt += f"\nObservation:{observation_text}\n\n"
        
        # Initialize messages
        self.messages = [{'role': 'user', 'content': start_prompt}]
        
        # Mark as visited
        self.visited_urls.add(self.root_url)
        
        # Get initial observation
        observation = self._get_observation(markdown, buttons, screenshot)
        
        return observation
    
    def step(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        The agent autonomously decides the action based on the task.
        
        Returns:
            observation: Current state observation
            reward: Reward signal
            done: Whether episode is complete
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        self.current_step += 1
        
        # Let agent run and collect responses
        step_info = {
            'step': self.current_step,
            'thought': None,
            'action': None,
            'observation': None,
            'memory_update': None,
            'final_answer': None
        }
        
        # Run agent for one iteration
        responses = self.agent.run(messages=self.messages, lang="en")
        
        current_observation = None
        for response_list in responses:
            if response_list and len(response_list) > 0:
                content = response_list[0].get("content", "")
                
                # Parse different types of responses
                if "\"}" in content and "Memory" not in content:
                    # This is a thought + action
                    step_info['thought'] = content.split("Action")[0] if "Action" in content else content
                    
                elif "\"}" in content and "Memory" in content:
                    # Memory update
                    memory_content = content[:-2] if content.endswith("\"}") else content
                    step_info['memory_update'] = memory_content
                    self.memory.append(memory_content)
                    
                elif "Final Answer" in content:
                    # Task completed
                    step_info['final_answer'] = content
                    self.done = True
                    break
                
                # Track latest observation
                current_observation = content
        
        # Store in trajectory
        self.trajectory.append(step_info)
        
        # Get new observation
        observation = self._get_current_observation()
        
        # Check if done (max steps or final answer reached)
        if self.current_step >= self.max_steps and not self.done:
            self.done = True
        
        # Calculate reward
        reward = self._calculate_reward(step_info, self.done)
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'step_info': step_info,
            'trajectory_length': len(self.trajectory),
            'task': self.task,
            'visited_urls': list(self.visited_urls),
            'memory': self.memory
        }
        
        return observation, reward, self.done, info
    
    def _extract_buttons(self, html: str) -> str:
        """Extract clickable buttons from HTML."""
        from bs4 import BeautifulSoup
        import re
        from utils import process_url
        
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        # Extract all clickable elements (a tags, buttons, etc.)
        for a_tag in soup.find_all('a', href=True):
            url = a_tag['href']
            text = ''.join(a_tag.stripped_strings)
            
            if text and "javascript" not in url and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
                full_url = process_url(self.root_url, url)
                if full_url.startswith(self.root_url):
                    links.append({'url': full_url, 'text': text})
        
        # Remove duplicates
        unique_links = {f"{item['url']}_{item['text']}": item for item in links}
        
        # Update button URL mapping
        for temp in list(unique_links.values()):
            self.button_url_mapping[temp["text"]] = temp["url"]
        
        # Save to file (required by WebWalker)
        with open("BUTTON_URL_ADIC.json", "w") as f:
            json.dump(self.button_url_mapping, f)
        
        # Format as button list
        info = ""
        for i in list(unique_links.values()):
            info += "<button>" + i["text"] + "</button>\n"
        
        return info
    
    def _get_observation(
        self, 
        markdown: str, 
        buttons: str, 
        screenshot: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current observation from the environment.
        
        Returns:
            Dictionary containing current state information
        """
        observation = {
            'url': self.current_url,
            'page_content': markdown,
            'available_buttons': buttons,
            'screenshot': screenshot,
            'step': self.current_step,
            'task': self.task,
            'memory': self.memory.copy(),
            'visited_urls': list(self.visited_urls)
        }
        
        return observation
    
    def _get_current_observation(self) -> Dict[str, Any]:
        """Get observation based on current state."""
        return {
            'url': self.current_url,
            'step': self.current_step,
            'task': self.task,
            'memory': self.memory.copy(),
            'visited_urls': list(self.visited_urls),
            'trajectory': self.trajectory.copy()
        }
    
    def _calculate_reward(
        self, 
        step_info: Dict[str, Any], 
        done: bool
    ) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            step_info: Information about the current step
            done: Whether episode is complete
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Small negative reward for each step (encourage efficiency)
        reward -= 0.1
        
        # Positive reward for memory updates (finding useful information)
        if step_info.get('memory_update'):
            reward += 1.0
        
        # Large positive reward if task completed successfully
        if done and step_info.get('final_answer'):
            reward += 10.0
        
        # Penalty if max steps reached without answer
        if done and not step_info.get('final_answer'):
            reward -= 5.0
        
        return reward
    
    def render(self, mode: str = 'human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
        """
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Task: {self.task}")
            print(f"Current URL: {self.current_url}")
            print(f"Visited URLs: {len(self.visited_urls)}")
            print(f"Memory Items: {len(self.memory)}")
            print(f"Done: {self.done}")
            
            if self.trajectory:
                last_step = self.trajectory[-1]
                if last_step.get('thought'):
                    print(f"\nLast Thought: {last_step['thought'][:100]}...")
                if last_step.get('memory_update'):
                    print(f"Memory Update: Yes")
                if last_step.get('final_answer'):
                    print(f"Final Answer: {last_step['final_answer'][:100]}...")
            print(f"{'='*60}\n")
    
    def close(self):
        """Clean up resources."""
        # Clean up temporary files
        if os.path.exists("ROOT_URL.txt"):
            os.remove("ROOT_URL.txt")
        if os.path.exists("BUTTON_URL_ADIC.json"):
            os.remove("BUTTON_URL_ADIC.json")
    
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get the current episode trajectory."""
        return self.trajectory.copy()
    
    def get_memory(self) -> List[str]:
        """Get the accumulated memory."""
        return self.memory.copy()


# Example usage
if __name__ == "__main__":
    # Setup LLM configuration
    llm_cfg = {
        'model': 'gpt-4',
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model_server': os.getenv('OPENAI_MODEL_SERVER', 'https://api.openai.com/v1'),
        'generate_cfg': {
            'top_p': 0.8,
            'max_input_tokens': 120000,
            'max_retries': 20
        },
    }
    
    # Create environment
    env = WebWalkerEnv(
        llm_cfg=llm_cfg,
        max_steps=10
    )
    
    # Reset environment with a task
    task = "When is the paper submission deadline for ACL 2025 Industry Track, and what is the venue specific address?"
    root_url = "https://2025.aclweb.org/"
    
    observation = env.reset(task=task, root_url=root_url)
    print(f"Initial observation URL: {observation['url']}")
    
    # Run episode
    done = False
    total_reward = 0
    
    while not done:
        # Agent acts autonomously
        observation, reward, done, info = env.step()
        total_reward += reward
        
        # Render current state
        env.render()
        
        print(f"Reward: {reward:.2f}, Total: {total_reward:.2f}")
    
    # Get trajectory and memory
    trajectory = env.get_trajectory()
    memory = env.get_memory()
    
    print(f"\n{'='*60}")
    print(f"Episode Summary:")
    print(f"Steps taken: {len(trajectory)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Memory items collected: {len(memory)}")
    print(f"URLs visited: {len(info['visited_urls'])}")
    
    if trajectory and trajectory[-1].get('final_answer'):
        print(f"\nFinal Answer: {trajectory[-1]['final_answer']}")
    
    # Clean up
    env.close()