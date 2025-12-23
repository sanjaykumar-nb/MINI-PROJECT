import os
import winshell
from win32com.client import Dispatch
from PIL import Image

def create_shortcut():
    # 1. Convert PNG to ICO if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "icon.png")
    ico_path = os.path.join(current_dir, "icon.ico")
    
    if os.path.exists(img_path):
        print(f"Converting {img_path} to icon...")
        img = Image.open(img_path)
        img.save(ico_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (32, 32)])
        print(f"Created {ico_path}")
    
    # 2. Create Desktop Shortcut
    desktop = winshell.desktop()
    path = os.path.join(desktop, "Research Assistant.lnk")
    target = os.path.join(current_dir, "desktop_app.py")
    # We want to run it with pythonw.exe to avoid a terminal window if possible, 
    # but for debugging python.exe is safer. User asked for an application feel.
    # Let's use python.exe for now to ensure they see errors if any.
    python_exe = os.path.join(os.environ['LOCALAPPDATA'], "Programs", "Python", "Python311", "python.exe") 
    # Fallback to sys.executable
    import sys
    python_exe = sys.executable
    
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(path)
    shortcut.Targetpath = python_exe
    shortcut.Arguments = f'"{target}"'
    shortcut.WorkingDirectory = current_dir
    shortcut.IconLocation = ico_path
    shortcut.save()
    
    print(f"Shortcut created at {path}")

if __name__ == "__main__":
    try:
        create_shortcut()
    except Exception as e:
        print(f"Error creating shortcut: {e}")
