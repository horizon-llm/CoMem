from typing import List, Dict, Any, Tuple
from .base import BaseMemory

class ConversationMemory(BaseMemory):
    """
    Memory manager for conversation history
    """
    def __init__(self):
        # data saves a list of conversation history for each environment
        # each conversation history is a list of dictionaries
        # where each dictionary contains the conversation record for one step
        # one step may contain fields like 'assistant', 'user', 'session', 'turn'
        self._data = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
    
    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of conversation history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'assistant', 'user', 'session', 'turn'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in record.keys()})
    
    def fetch(self, history_length: int = None) -> List[List[Dict[str, Any]]]:
        """
        Fetch and format recent conversation history for each environment instance.

        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.

        Returns:
            List[List[Dict[str, Any]]]:
                A list of conversation histories for each environment,
                where each history is a list of dictionaries containing the conversation records.
                It may contain a list of dictionaries with keys like 'assistant', 'user', etc.
        """
        memory_contexts = []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:] if history_length is not None else self._data[env_idx]
            memory_contexts.append(recent)

        return memory_contexts