"""
Shared app state: task registries and locks.
Same idea as DFAPI state.py - keeps main.py and manager in sync.
"""

import threading

# guids currently being ingested or answered (prevents overload)
processing_files = set()
processing_lock = threading.Lock()

MAX_CONCURRENT_TASKS = 2
