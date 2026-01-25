import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import csv
from pathlib import Path

# 設置路徑
images_dir = r"./dogs-vs-cats/test1/"
output_csv = r"./labels_result.csv"

class ImageLabelingApp:
    def __init__(self, root, images_dir, output_csv):
        self.root = root
        self.root.title("圖片標註工具 - 貓與狗")
        self.root.geometry("800x700")
        
        self.images_dir = images_dir
        self.output_csv = output_csv
        
        # 獲取所有圖像文件
        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
        ])
        
        self.current_index = 0
        self.labels = {}  # 儲存標籤: {filename: label}
        
        # 讀取已存在的標註結果
        self.load_from_csv()
        
        # 找到第一個未標註的圖像
        self.find_next_unlabeled()
        
        # 創建UI元件
        self.setup_ui()
        self.show_image()
    
    def setup_ui(self):
        """設置用戶界面"""
        # 標題
        title_label = tk.Label(self.root, text="圖片標註工具 - 貓與狗", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 快捷鍵說明
        shortcut_label = tk.Label(
            self.root, 
            text="快捷鍵: 0/C=貓  1/D=狗  ←/→=上/下一張",
            font=("Arial", 9),
            fg="gray"
        )
        shortcut_label.pack(pady=2)
        
        # 進度信息
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.progress_label.pack(pady=5)
        
        # 圖像顯示框架
        image_frame = tk.Frame(self.root, bg="white", border=2, relief=tk.SUNKEN)
        image_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(image_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 綁定鍵盤快捷鍵
        self.root.bind('0', lambda e: self.label_image(0))  # 0 = 貓
        self.root.bind('c', lambda e: self.label_image(0))  # c = 貓
        self.root.bind('1', lambda e: self.label_image(1))  # 1 = 狗
        self.root.bind('d', lambda e: self.label_image(1))  # d = 狗
        self.root.bind('<Left>', lambda e: self.prev_image())  # 左箭頭 = 上一張
        self.root.bind('<Right>', lambda e: self.next_image())  # 右箭頭 = 下一張
        
        # 文件名顯示
        self.filename_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.filename_label.pack(pady=5)
        
        # 按鈕框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # 貓按鈕 (label=0)
        self.cat_button = tk.Button(
            button_frame, 
            text="🐱 貓 (0)", 
            command=lambda: self.label_image(0),
            width=15,
            height=2,
            font=("Arial", 12, "bold"),
            bg="#FFB6C1",
            activebackground="#FF69B4"
        )
        self.cat_button.pack(side=tk.LEFT, padx=10)
        
        # 狗按鈕 (label=1)
        self.dog_button = tk.Button(
            button_frame, 
            text="🐕 狗 (1)", 
            command=lambda: self.label_image(1),
            width=15,
            height=2,
            font=("Arial", 12, "bold"),
            bg="#87CEEB",
            activebackground="#4169E1"
        )
        self.dog_button.pack(side=tk.LEFT, padx=10)
        
        # 控制框架
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # 上一張按鈕
        self.prev_button = tk.Button(
            control_frame,
            text="← 上一張",
            command=self.prev_image,
            width=10
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        # 下一張按鈕
        self.next_button = tk.Button(
            control_frame,
            text="下一張 →",
            command=self.next_image,
            width=10
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # 保存按鈕
        self.save_button = tk.Button(
            control_frame,
            text="💾 保存結果",
            command=self.save_results,
            width=10,
            bg="#90EE90",
            activebackground="#228B22"
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
    
    def show_image(self):
        """顯示當前圖像"""
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("完成", "所有圖像已標註！")
            return
        
        filename = self.image_files[self.current_index]
        filepath = os.path.join(self.images_dir, filename)
        
        try:
            # 打開並調整圖像大小
            image = Image.open(filepath)
            image.thumbnail((700, 500), Image.LANCZOS)
            
            # 轉換為PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # 更新標籤
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持參考
            
            # 更新文件名標籤
            file_id = os.path.splitext(filename)[0]
            self.filename_label.config(text=f"文件: {filename} (ID: {file_id})")
            
            # 更新進度
            progress_text = f"進度: {self.current_index + 1} / {len(self.image_files)}"
            if filename in self.labels:
                label_text = "貓" if self.labels[filename] == 0 else "狗"
                progress_text += f" [已標註為: {label_text}]"
            self.progress_label.config(text=progress_text)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"無法打開圖像: {e}")
    
    def label_image(self, label):
        """標註當前圖像"""
        if self.current_index >= len(self.image_files):
            return
        
        filename = self.image_files[self.current_index]
        self.labels[filename] = label
        
        # 自動保存到CSV
        self.save_to_csv()
        
        # 自動跳到下一張
        if self.current_index < len(self.image_files) - 1:
            self.next_image()
    
    def next_image(self):
        """顯示下一張圖像"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()
        else:
            messagebox.showinfo("提示", "已到達最後一張圖像")
    
    def prev_image(self):
        """顯示上一張圖像"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()
        else:
            messagebox.showinfo("提示", "已到達第一張圖像")
    
    def load_from_csv(self):
        """從CSV文件讀取已標註的結果"""
        if not os.path.exists(self.output_csv):
            print(f"CSV文件不存在: {self.output_csv}")
            return
        
        try:
            with open(self.output_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        file_id = str(row['id']).strip()
                        label = int(row['label'])
                        
                        # 根據ID查找對應的文件名
                        found = False
                        for filename in self.image_files:
                            # 獲取文件名不含副檔名
                            name_without_ext = os.path.splitext(filename)[0]
                            if name_without_ext == file_id:
                                self.labels[filename] = label
                                print(f"已載入: {filename} -> {label}")
                                found = True
                                break
                        if not found:
                            print(f"警告: 找不到對應的文件: ID={file_id}")
                    except (ValueError, KeyError) as e:
                        print(f"行讀取錯誤: {e}, row={row}")
                        continue
            print(f"成功讀取CSV，已載入 {len(self.labels)} 個標註")
        except Exception as e:
            print(f"讀取CSV失敗: {e}")
    
    def find_next_unlabeled(self):
        """找到第一個未標註的圖像"""
        for i, filename in enumerate(self.image_files):
            if filename not in self.labels:
                self.current_index = i
                print(f"找到第一個未標註的圖像: {filename} (索引 {i})")
                return
        # 如果所有圖像都已標註
        print("所有圖像已標註完成！")
        self.current_index = 0
    
    def save_to_csv(self):
        """自動保存標註結果為CSV文件"""
        try:
            with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'label'])
                
                # 只寫入已標註的項目，避免重複
                written_ids = set()
                for filename in self.image_files:
                    if filename in self.labels:
                        # 從文件名中提取ID (例如: 1.jpg -> 1)
                        file_id = os.path.splitext(filename)[0]
                        if file_id not in written_ids:
                            label = self.labels[filename]
                            writer.writerow([file_id, label])
                            written_ids.add(file_id)
        except Exception as e:
            messagebox.showerror("錯誤", f"保存失敗: {e}")
    
    def save_results(self):
        """保存標註結果為CSV文件"""
        if not self.labels:
            messagebox.showwarning("警告", "還沒有任何標註！")
            return
        
        self.save_to_csv()
        
        messagebox.showinfo(
            "成功",
            f"結果已保存！\n\n"
            f"文件路徑: {self.output_csv}\n"
            f"已標註: {len(self.labels)} / {len(self.image_files)}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingApp(root, images_dir, output_csv)
    root.mainloop()
