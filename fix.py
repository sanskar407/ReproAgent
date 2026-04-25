import os
import glob
from reprlib import repr
import traceback

# 1. Clean corrupted json files
files = glob.glob("data/papers/**/*.json", recursive=True)
for f in files:
    try:
        if os.path.getsize(f) == 0:
            os.remove(f)
            print(f"Removed corrupted empty file: {f}")
    except:
        pass

# 2. Re-create sample papers
try:
    from reproagent.papers import create_sample_papers
    create_sample_papers()
    print("Sample papers re-created.")
except Exception as e:
    print(f"Failed to create sample papers: {e}")

# 3. Test environment
try:
    from reproagent.environment import ReproAgentEnv
    env = ReproAgentEnv(difficulty='easy', max_steps=10, use_llm=False)
    print('SUCCESS')
except Exception as e:
    print('FULL ERROR:')
    traceback.print_exc()
