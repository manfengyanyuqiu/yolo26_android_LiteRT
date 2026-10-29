"""Desktop UI for YOLOv5 / YOLOv8 / YOLO26 -> MediaTek TFLite."""
from datetime import datetime
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from model_exporter import ROOT, RUNTIME, parser

class Exporter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('FYDNN 模型导出工具 0.6.3')
        self.geometry('960x760'); self.minsize(800,620)
        self.events = queue.Queue(); self.busy = False
        defaults = parser().parse_args(['--output',str(ROOT/'exports')])
        self.family = tk.StringVar(value='26')
        self.model = tk.StringVar(value='s')
        self.size = tk.StringVar(value='320')
        self.weights = tk.StringVar()
        self.calibration = tk.StringVar(value=str(defaults.calibration_dir))
        self.image = tk.StringVar()
        self.output = tk.StringVar(value=str(ROOT/'exports'/('yolo26s-320-'+datetime.now().strftime('%Y%m%d-%H%M%S'))))
        self.formats = {name:tk.BooleanVar(value=True) for name in ('fp16','int8','fp32')}
        self.evaluate = tk.BooleanVar(value=True)
        self.state = tk.StringVar(value='就绪：YOLOv5 / YOLOv8 / YOLO26，固定 batch=1。')
        self.inputs=[]
        frame=ttk.Frame(self,padding=16); frame.pack(fill='both',expand=True)
        frame.columnconfigure(1,weight=1); frame.rowconfigure(11,weight=1)
        ttk.Label(frame,text='YOLOv5 / YOLOv8 / YOLO26 → 天玑 NPU',font=('Microsoft YaHei UI',16,'bold')).grid(row=0,column=0,columnspan=3,sticky='w',pady=(0,12))
        ttk.Label(frame,text='复用 Ultralytics / onnx2tf / TensorFlow；执行四维输出、坐标归一化和 Attention 布局修正。').grid(row=1,column=0,columnspan=3,sticky='w',pady=(0,12))
        options=ttk.Frame(frame); options.grid(row=2,column=0,columnspan=3,sticky='ew',pady=5)
        ttk.Label(options,text='系列：').pack(side='left')
        family=ttk.Combobox(options,textvariable=self.family,values=['v5','v8','26'],state='readonly',width=5); family.pack(side='left',padx=(3,12))
        ttk.Label(options,text='规格：').pack(side='left')
        model=ttk.Combobox(options,textvariable=self.model,values=['n','s'],state='readonly',width=5); model.pack(side='left',padx=(3,18))
        ttk.Label(options,text='输入尺寸：').pack(side='left')
        size=ttk.Combobox(options,textvariable=self.size,values=['224','320','416','640'],width=8); size.pack(side='left',padx=(3,10))
        ttk.Label(options,text='× 同尺寸，须为 32 的倍数').pack(side='left')
        self.inputs.extend([(family,'readonly'),(model,'readonly'),(size,'normal')])
        self.path_row(frame,3,'源模型 .onnx / .pt',self.weights,'weights')
        self.path_row(frame,4,'校准图片目录',self.calibration,'directory')
        self.path_row(frame,5,'验证图片（可选）',self.image,'image')
        self.path_row(frame,6,'输出目录（空目录）',self.output,'directory')
        ttk.Label(frame,text='v5：v7 检测头 ONNX；v8/26：PT 或 ONNX。动态图固定尺寸并校验；INT8 需 ≥32 张真实图片。').grid(row=7,column=0,columnspan=3,sticky='w',pady=5)
        formats=ttk.Frame(frame); formats.grid(row=8,column=0,columnspan=3,sticky='w',pady=8)
        ttk.Label(formats,text='导出精度：').pack(side='left')
        for name,label in [('fp16','FP16 权重 / FP32 I/O'),('int8','全 INT8（逐张量）'),('fp32','FP32')]:
            check=ttk.Checkbutton(formats,text=label,variable=self.formats[name]); check.pack(side='left',padx=8); self.inputs.append((check,'normal'))
        evaluate=ttk.Checkbutton(frame,text='运行主机 COCO 精度评估（校准目录需配套 labels/train2017）',variable=self.evaluate)
        evaluate.grid(row=9,column=0,columnspan=3,sticky='w'); self.inputs.append((evaluate,'normal'))
        bar=ttk.Frame(frame); bar.grid(row=10,column=0,columnspan=3,sticky='ew',pady=10)
        self.start=ttk.Button(bar,text='开始导出',command=self.start_export); self.start.pack(side='left')
        ttk.Button(bar,text='打开输出目录',command=self.open_output).pack(side='left',padx=12)
        ttk.Label(bar,textvariable=self.state).pack(side='left')
        self.log=scrolledtext.ScrolledText(frame,wrap='word',font=('Consolas',10),state='disabled'); self.log.grid(row=11,column=0,columnspan=3,sticky='nsew')
        ttk.Label(frame,text='完成表示主机导出与所选校验成功；NPU 编译、执行和实际速度须在手机独立验证。').grid(row=12,column=0,columnspan=3,sticky='w',pady=(8,0))
        self.protocol('WM_DELETE_WINDOW',self.close_window); self.after(100,self.drain)
    def path_row(self,parent,row,label,variable,kind):
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky='w',padx=(0,10),pady=5)
        entry=ttk.Entry(parent,textvariable=variable); entry.grid(row=row,column=1,sticky='ew',pady=5)
        button=ttk.Button(parent,text='选择…',command=lambda:self.browse(variable,kind)); button.grid(row=row,column=2,padx=(8,0))
        self.inputs.extend([(entry,'normal'),(button,'normal')])
    def browse(self,variable,kind):
        if kind=='directory': path=filedialog.askdirectory(parent=self,mustexist=False)
        else: path=filedialog.askopenfilename(parent=self,filetypes=[('YOLO models','*.onnx *.pt'),('ONNX','*.onnx'),('PyTorch','*.pt')] if kind=='weights' else [('Images','*.jpg *.jpeg *.png')])
        if path: variable.set(path)
    def append(self,message):
        self.log.configure(state='normal'); self.log.insert('end',message); self.log.see('end'); self.log.configure(state='disabled')
    def start_export(self):
        try:
            size=int(self.size.get()); formats=[name for name,v in self.formats.items() if v.get()]
            if size<32 or size%32: raise ValueError('尺寸须为正整数且为 32 的倍数。')
            if not formats: raise ValueError('至少选择一种精度。')
            out=Path(self.output.get()).resolve()
            if not self.output.get().strip(): raise ValueError('请选择输出目录。')
            if out.exists() and (not out.is_dir() or any(out.iterdir())): raise ValueError('输出目录须为空，避免覆盖已有模型。')
            source=self.weights.get().strip()
            if self.family.get()=='v5' and (not source or Path(source).suffix.lower()!='.onnx'):
                raise ValueError('YOLOv5 请选择 v7 检测头导出的 .onnx。')
            if source and Path(source).suffix.lower() not in ('.pt','.onnx'):
                raise ValueError('源模型请选择 .pt 或 .onnx。')
            args=['--family',self.family.get(),'--model',self.model.get(),'--size',str(size),'--formats',*formats,'--output',str(out),
                  '--calibration-dir',self.calibration.get()]
            if source: args+=['--onnx' if Path(source).suffix.lower()=='.onnx' else '--weights',source]
            if self.image.get().strip(): args+=['--test-image',self.image.get()]
            if self.evaluate.get(): args+=['--evaluate']
            runtime=Path(RUNTIME.get('convert_python',ROOT/'.toolchain/model-env/Scripts/python.exe'))
            if not runtime.is_file(): raise ValueError('转换环境缺失，请核对 exporter-runtime.json 的 Python 路径。')
        except (ValueError,OSError) as error:
            messagebox.showerror('导出参数',str(error),parent=self); return
        self.busy=True; self.start.configure(state='disabled'); self.state.set('导出中…')
        for widget,_ in self.inputs: widget.configure(state='disabled')
        self.append('\n开始导出 YOLO'+self.family.get()+self.model.get()+' / '+str(size)+'×'+str(size)+'\n')
        threading.Thread(target=self.worker,args=(runtime,args),daemon=True).start()
    def worker(self,runtime,args):
        try:
            env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
            with subprocess.Popen([str(runtime),str(ROOT/'tools/model_exporter.py'),*args],cwd=ROOT,env=env,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0) as process:
                for line in process.stdout: self.events.put(('log',line))
                code=process.wait()
            self.events.put(('done',code))
        except Exception as error:
            self.events.put(('log',str(error)+'\n')); self.events.put(('done',1))
    def drain(self):
        try:
            for _ in range(250):
                kind,value=self.events.get_nowait()
                if kind=='log': self.append(value)
                else:
                    self.busy=False; self.start.configure(state='normal')
                    for widget,state in self.inputs: widget.configure(state=state)
                    self.state.set('导出完成；见输出目录。' if value==0 else '导出失败；查看日志定位失败阶段。')
        except queue.Empty: pass
        self.after(100,self.drain)
    def open_output(self):
        path=Path(self.output.get())
        if path.is_dir(): os.startfile(str(path))
        else: messagebox.showinfo('输出目录','目录将在开始导出后创建。',parent=self)
    def close_window(self):
        if self.busy:
            messagebox.showinfo('导出中','导出任务正在运行，请等待任务完成后关闭窗口。',parent=self)
        else: self.destroy()

if __name__=='__main__':
    app=Exporter()
    if '--smoke-test' in sys.argv:
        app.update(); app.destroy(); print('GUI_SMOKE_TEST_PASS')
    else: app.mainloop()
