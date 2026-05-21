from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from pathlib import Path
from datasets import load_dataset
import os

# Download the Lite version of the dataset
ds = load_dataset("R2E-Gym/R2E-Gym-Lite")
split = "train"
env_index = 0 # Try any index between 0 and len(ds[split])

env_args = EnvArgs(ds=ds[split][env_index])

env = RepoEnv(env_args, backend='kubernetes')
agent_args = AgentArgs.from_yaml(
Path("./src/r2egym/agenthub/config/r2egym/edit_fn_calling.yaml")
)
os.environ["AWS_REGION_NAME"] = "us-east-2"
agent_args.llm_name = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0" #"claude-3-5-sonnet-20241022" # Use any supported model

agent = Agent(name="EditingAgent", args=agent_args)
output = agent.run(env, max_steps=40, use_fn_calling=True)

print("Trajectory Output:")
print(output)
