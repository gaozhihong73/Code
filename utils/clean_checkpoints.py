import os
import shutil
import sys


def find_checkpoint_dirs(root_dir, dry_run=True):
    """遍历目录，收集所有 .ipynb_checkpoints 路径"""
    matches = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if ".ipynb_checkpoints" in dirnames:
            full_path = os.path.join(dirpath, ".ipynb_checkpoints")
            matches.append(full_path)
    return matches


def main():
    """扫描并清理目录树中的所有 .ipynb_checkpoints 目录。"""
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    dry_run = True  # 默认先预览

    # 检查是否有 -y 或 --yes 参数来跳过确认
    if "-y" in sys.argv or "--yes" in sys.argv:
        dry_run = False

    print(f"🔍 正在扫描目录: {os.path.abspath(target_dir)}")
    dirs_to_delete = find_checkpoint_dirs(target_dir, dry_run)

    if not dirs_to_delete:
        print("✅ 没有找到任何 .ipynb_checkpoints 目录。")
        return

    print(f"\n📁 找到 {len(dirs_to_delete)} 个目录：")
    for d in dirs_to_delete:
        print(f"  - {d}")

    if dry_run:
        print("\n⚠️ 当前为试运行模式，未执行删除。")
        print("💡 如需真正删除，请加上 -y 或 --yes 参数，例如：")
        print(f"   python clean_checkpoints.py {target_dir} -y")
        return

    # 真正删除
    confirm = input(f"\n⚠️ 确认要删除以上 {len(dirs_to_delete)} 个目录吗？(y/N): ")
    if confirm.lower() != "y":
        print("❌ 已取消操作。")
        return

    for d in dirs_to_delete:
        try:
            shutil.rmtree(d)
            print(f"✅ 已删除: {d}")
        except Exception as e:
            print(f"❌ 删除失败 {d}: {e}")


if __name__ == "__main__":
    main()
