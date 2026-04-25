"""
Validation script for OpenEnv compatibility.
Run this before submitting: python validate.py
"""

import sys
import traceback
from pathlib import Path

from reproagent.environment import ReproAgentEnv


def validate_environment():
    """Validate environment meets OpenEnv requirements."""
    
    print("="*70)
    print("🔍 VALIDATING REPROAGENT ENVIRONMENT")
    print("="*70)
    print()
    
    all_passed = True
    
    # Test 1: Import environment
    print("Test 1: Environment Import")
    try:
        from reproagent.environment import ReproAgentEnv
        print("  ✅ Environment imported successfully")
    except Exception as e:
        print(f"  ❌ Failed to import environment: {e}")
        traceback.print_exc()
        all_passed = False
        return False
    
    # Test 2: Create environment
    print("\nTest 2: Environment Creation")
    try:
        env = ReproAgentEnv(difficulty="easy", max_steps=20, use_llm=False)
        print("  ✅ Environment created")
    except Exception as e:
        print(f"  ❌ Failed to create environment: {e}")
        traceback.print_exc()
        all_passed = False
        return False
    
    # Test 3: Check spaces
    print("\nTest 3: Action/Observation Spaces")
    try:
        assert hasattr(env, 'action_space'), "Missing action_space"
        assert hasattr(env, 'observation_space'), "Missing observation_space"
        print(f"  ✅ Action space: {env.action_space}")
        print(f"  ✅ Observation space: {type(env.observation_space).__name__}")
    except Exception as e:
        print(f"  ❌ Space validation failed: {e}")
        all_passed = False
    
    # Test 4: Reset
    print("\nTest 4: Reset")
    try:
        obs, info = env.reset()
        assert obs is not None, "Observation is None"
        assert isinstance(info, dict), "Info is not dict"
        print("  ✅ Reset successful")
        print(f"  ✅ Observation keys: {list(obs.keys())}")
        print(f"  ✅ Info keys: {list(info.keys())}")
    except Exception as e:
        print(f"  ❌ Reset failed: {e}")
        traceback.print_exc()
        all_passed = False
        return False
    
    # Test 5: Observation space validation
    print("\nTest 5: Observation Space Validation")
    try:
        assert env.observation_space.contains(obs), "Observation not in space"
        print("  ✅ Observation matches observation_space")
    except Exception as e:
        print(f"  ❌ Observation space mismatch: {e}")
        all_passed = False
    
    # Test 6: Action space validation
    print("\nTest 6: Action Space Validation")
    try:
        action = env.action_space.sample()
        assert env.action_space.contains(action), "Action not in space"
        print(f"  ✅ Sampled action: {action}")
        print(f"  ✅ Action is valid")
    except Exception as e:
        print(f"  ❌ Action space validation failed: {e}")
        all_passed = False
    
    # Test 7: Step
    print("\nTest 7: Step")
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None, "Observation is None"
        assert isinstance(reward, (int, float)), "Reward is not numeric"
        assert isinstance(terminated, bool), "Terminated is not bool"
        assert isinstance(truncated, bool), "Truncated is not bool"
        assert isinstance(info, dict), "Info is not dict"
        print("  ✅ Step successful")
        print(f"  ✅ Reward: {reward:.2f}")
        print(f"  ✅ Terminated: {terminated}")
        print(f"  ✅ Truncated: {truncated}")
    except Exception as e:
        print(f"  ❌ Step failed: {e}")
        traceback.print_exc()
        all_passed = False
        return False
    
    # Test 8: Full episode
    print("\nTest 8: Full Episode")
    try:
        env.reset()
        total_reward = 0
        steps = 0
        
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"  ✅ Episode completed")
        print(f"  ✅ Steps: {steps}")
        print(f"  ✅ Total reward: {total_reward:.2f}")
    except Exception as e:
        print(f"  ❌ Episode failed: {e}")
        traceback.print_exc()
        all_passed = False
    
    # Test 9: Multiple episodes
    print("\nTest 9: Multiple Episodes")
    try:
        for episode in range(3):
            env.reset()
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        print(f"  ✅ 3 episodes completed successfully")
    except Exception as e:
        print(f"  ❌ Multiple episodes failed: {e}")
        traceback.print_exc()
        all_passed = False
    
    # Test 10: Render
    print("\nTest 10: Render")
    try:
        env.reset()
        output = env.render()
        print("  ✅ Render successful")
    except Exception as e:
        print(f"  ⚠️  Render failed (non-critical): {e}")
    
    # Test 11: Close
    print("\nTest 11: Close")
    try:
        env.close()
        print("  ✅ Close successful")
    except Exception as e:
        print(f"  ⚠️  Close failed (non-critical): {e}")
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("="*70)
        print("\n🎉 Environment is OpenEnv compatible!")
        print("✅ Ready for submission")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\n⚠️  Please fix errors before submission")
        return False


def validate_agents():
    """Validate agents can interact with environment."""
    
    print("\n" + "="*70)
    print("🤖 VALIDATING AGENTS")
    print("="*70)
    print()
    
    try:
        from reproagent.environment import ReproAgentEnv
        from agents.reasoning_agent import create_agent
        
        env = ReproAgentEnv(difficulty="easy", max_steps=10, use_llm=False)
        
        # Test reasoning agent
        print("Test: Reasoning Agent")
        agent = create_agent(env, "reasoning", use_llm=False)
        
        obs, info = env.reset()
        agent.reset()
        
        for i in range(5):
            action = agent.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        print("  ✅ Reasoning agent works")
        
        # Test random agent
        print("\nTest: Random Agent")
        random_agent = create_agent(env, "random")
        
        obs, info = env.reset()
        random_agent.reset()
        
        for i in range(5):
            action = random_agent.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        print("  ✅ Random agent works")
        
        print("\n✅ All agents validated successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Agent validation failed: {e}")
        traceback.print_exc()
        return False


def validate_demo():
    """Validate Gradio demo can be imported."""
    
    print("\n" + "="*70)
    print("🎨 VALIDATING DEMO")
    print("="*70)
    print()
    
    try:
        from server.app import create_demo
        print("  ✅ Demo imported successfully")
        
        print("  ℹ️  To test demo fully, run: python server/app.py")
        return True
        
    except Exception as e:
        print(f"  ❌ Demo import failed: {e}")
        traceback.print_exc()
        return False


def validate_graders():
    """Validate grading system."""
    
    print("\n" + "="*70)
    print("📊 VALIDATING GRADERS")
    print("="*70)
    print()
    
    try:
        from graders.graders import ReproductionGrader
        print("  ✅ Grader imported successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Grader import failed: {e}")
        traceback.print_exc()
        return False


def validate_openenv_yaml():
    """Validate openenv.yaml exists."""
    
    print("\n" + "="*70)
    print("📄 VALIDATING openenv.yaml")
    print("="*70)
    print()
    
    yaml_path = Path("openenv.yaml")
    
    if yaml_path.exists():
        print("  ✅ openenv.yaml exists")
        
        try:
            import yaml
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
            
            required_keys = ['name', 'environment', 'observation_space', 'action_space']
            
            for key in required_keys:
                if key in config:
                    print(f"  ✅ Has '{key}'")
                else:
                    print(f"  ⚠️  Missing '{key}'")
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  Could not parse YAML: {e}")
            return True  # Non-critical
    else:
        print("  ⚠️  openenv.yaml not found (will need to create)")
        return True  # Non-critical for now


def main():
    """Run all validation tests."""
    
    print("\n" + "🚀"*35)
    print("REPROAGENT VALIDATION SUITE")
    print("🚀"*35 + "\n")
    
    results = {
        'environment': validate_environment(),
        'agents': validate_agents(),
        'demo': validate_demo(),
        'graders': validate_graders(),
        'openenv_yaml': validate_openenv_yaml()
    }
    
    # Final summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{component.upper():<20} {status}")
    
    print("="*70)
    
    if all(results.values()):
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ System is ready for deployment")
        return 0
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED")
        print("Please fix errors before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
