import subprocess, time, os

max_used = 0

while True:
    try:
        # Chạy rocm-smi và lọc dòng chứa "VRAM Total Used Memory"
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"],
            text=True
        )
        for line in out.splitlines():
            if "VRAM Total Used Memory" in line:
                used = int(line.split()[-1])  # đơn vị Bytes
                break
        else:
            used = 0

        if used > max_used:
            max_used = used

        used_gb = used / 1024**3
        max_gb = max_used / 1024**3

        os.system("clear")
        print(f"VRAM đang dùng : {used_gb:.2f} GB")
        print(f"VRAM lớn nhất : {max_gb:.2f} GB")

    except Exception as e:
        print("Lỗi:", e)

    time.sleep(0.5)
