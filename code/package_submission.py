import zipfile
import shutil
from pathlib import Path

def package():
    root = Path(__file__).resolve().parents[1]
    submission_dir = root / "submission_artifacts"
    submission_dir.mkdir(exist_ok=True)

    # 1. Copy output.csv
    src_output = root / "output.csv"
    dest_output = submission_dir / "output.csv"
    if src_output.exists():
        shutil.copy2(src_output, dest_output)
        print(f"Copied output.csv to {dest_output}")
    else:
        print("[Error] output.csv not found at root!")

    # 2. Copy chat transcript log.txt
    log_path = Path.home() / "hackerrank_orchestrate" / "log.txt"
    dest_log = submission_dir / "chat_transcript_log.txt"
    if log_path.exists():
        shutil.copy2(log_path, dest_log)
        print(f"Copied chat transcript log from {log_path} to {dest_log}")
    else:
        print("[Error] Chat transcript log.txt not found!")

    # 3. Create code.zip containing the code/ directory
    zip_path = submission_dir / "code.zip"
    code_dir = root / "code"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in code_dir.rglob('*'):
            if '__pycache__' in file_path.parts:
                continue
            if file_path.suffix in ('.pyc', '.pyo'):
                continue
            if file_path.is_file():
                arcname = file_path.relative_to(root)
                zipf.write(file_path, arcname)
    print(f"Created code.zip at {zip_path}")

if __name__ == "__main__":
    package()
