from pathlib import Path
import contextlib,io,os,sys,base64
import json
os.environ['MPLBACKEND']='Agg'
root=Path(__file__).resolve().parent
selected=set(sys.argv[1:])
for nb in sorted((root/'notebooks').glob('*.ipynb')):
    if selected and nb.stem not in selected:
        continue
    print('Running',nb.name,flush=True)
    notebook=json.loads(nb.read_text(encoding='utf-8'))
    old_dir=Path.cwd()
    os.chdir(nb.parent)
    scope={'__name__':'__main__','display':print}
    try:
        for number,cell in enumerate(notebook['cells'],start=1):
            if cell['cell_type']!='code':
                continue
            output=io.StringIO()
            figure_dir=root/'outputs/figures'
            before={p.name:p.stat().st_mtime_ns for p in figure_dir.glob('*.png')} if figure_dir.exists() else {}
            with contextlib.redirect_stdout(output):
                exec(compile(''.join(cell['source']),f'{nb.name} cell {number}','exec'),scope)
            cell['execution_count']=number
            captured=output.getvalue()
            cell_outputs=[{'output_type':'stream','name':'stdout','text':captured}] if captured else []
            if figure_dir.exists():
                for image in sorted(figure_dir.glob('*.png')):
                    if image.name not in before or image.stat().st_mtime_ns!=before[image.name]:
                        cell_outputs.append({'output_type':'display_data','data':{'image/png':base64.b64encode(image.read_bytes()).decode('ascii')},'metadata':{}})
            cell['outputs']=cell_outputs
            print('  finished code cell',number,flush=True)
    finally:
        os.chdir(old_dir)
    nb.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
