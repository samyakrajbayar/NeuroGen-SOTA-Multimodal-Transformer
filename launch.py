#!/usr/bin/env python3
"""
Air Writing System Launcher
===========================

Simple launcher script with setup checks and helpful messages.

Usage:
    python launch.py              # Launch with default settings
    python launch.py --setup      # Run setup and install dependencies
    python launch.py --basic      # Launch basic version
    python launch.py --advanced   # Launch advanced version (default)
    python launch.py --check      # Check system requirements
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          ✋ ADVANCED AIR WRITING SYSTEM 🤚                ║
    ║                                                           ║
    ║           AI-Powered Gesture Recognition                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)
    
    # Optional dependencies
    optional = {
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
    }
    
    print("\nOptional packages:")
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} (optional)")
    
    return missing


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_camera():
    """Check if camera is available."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera detected")
                return True
        print("❌ No camera detected")
        return False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False


def run_system_check():
    """Run complete system check."""
    print("\n🔍 Running system check...")
    print("=" * 50)
    
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": len(check_dependencies()) == 0,
        "Camera": check_camera(),
    }
    
    print("=" * 50)
    
    if all(checks.values()):
        print("\n✅ All checks passed! Ready to launch.")
        return True
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return False


def launch_application(advanced=True):
    """Launch the air writing application."""
    if advanced:
        print("\n🚀 Launching Advanced Air Writing System...")
        script = "air_writing_advanced.py"
    else:
        print("\n🚀 Launching Basic Air Writing System...")
        script = "air_writing_basic.py"
    
    if not os.path.exists(script):
        print(f"❌ Error: {script} not found!")
        return False
    
    try:
        subprocess.run([sys.executable, script])
        return True
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return True
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Air Writing System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py              # Launch advanced version
  python launch.py --basic      # Launch basic version
  python launch.py --setup      # Install dependencies
  python launch.py --check      # Check system requirements
        """
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Install dependencies and setup environment'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check system requirements'
    )
    
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Launch basic version'
    )
    
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Launch advanced version (default)'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check Python version first
    if not check_python_version():
        sys.exit(1)
    
    # Handle different modes
    if args.setup:
        print("\n🔧 Setup Mode")
        if install_dependencies():
            run_system_check()
    
    elif args.check:
        run_system_check()
    
    else:
        # Launch mode
        if not run_system_check():
            print("\n💡 Run 'python launch.py --setup' to install missing dependencies")
            response = input("\nTry to launch anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Launch appropriate version
        advanced = not args.basic
        launch_application(advanced)


if __name__ == "__main__":
    main()
