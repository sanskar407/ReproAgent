"""Quick test of the demo."""

# Test imports
try:
    from server.app import create_demo
    print("✅ Demo imports successful")
    
    # Create demo
    demo = create_demo()
    print("✅ Demo created successfully")
    
    print("\n🎉 Demo is ready!")
    print("Run: python server/app.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
