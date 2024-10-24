from PIL import Image
import imagehash
import os
from rich.console import Console
from rich.progress import Progress

console = Console()

def remove_duplicates(file_paths, folder_path, similarity_threshold):
    unique_images = []
    hashes = {}

    with Progress() as progress:
        task = progress.add_task("[cyan]去重处理中...[/cyan]", total=len(file_paths))

        for file_path in file_paths:
            try:
                img = Image.open(file_path)
                img_hash = imagehash.average_hash(img)
                
                is_duplicate = False
                for existing_hash, existing_path in hashes.items():
                    if (img_hash - existing_hash) < (100 - similarity_threshold):
                        is_duplicate = True
                        console.print(f"[yellow]检测到相似图片: {file_path} 与 {existing_path}[/yellow]")
                        break

                if not is_duplicate:
                    hashes[img_hash] = file_path
                    unique_images.append(img)
                    console.print(f"[green]添加新图片: {file_path}[/green]")
                else:
                    console.print(f"[yellow]跳过重复图片: {file_path}[/yellow]")

            except Exception as e:
                console.print(f"[red]处理 {file_path} 时出错: {str(e)}[/red]")

            progress.update(task, advance=1)

    console.print(f"[bold green]去重完成。原始图片数: {len(file_paths)}, 去重后图片数: {len(unique_images)}[/bold green]")
    return unique_images

def get_files_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
