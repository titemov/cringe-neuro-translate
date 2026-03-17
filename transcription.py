import os

print("Enter filename")
file = str(input())

os.system(f"whisper {file} --language English --model tiny --device cuda")

print("done!")